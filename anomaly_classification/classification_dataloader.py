"""GI classification data loader using tf.data.
Expects a root directory with one subdirectory per class:

    image_dir/
        class_a/image1.jpg ...
        class_b/image1.jpg ...
        ...

Streams images from disk batch-by-batch — no full-dataset RAM allocation.
"""
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List
import tensorflow as tf


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    image_dir:  str = "../Dataset/kvasir-dataset-v2/"
    img_size:   Tuple[int, int] = (224, 224)
    batch_size: int = 8
    train_frac: float = 0.70
    val_frac:   float = 0.20          # remainder becomes test
    seed:       int = 42
    image_ext:  str = ".jpg"
    num_classes: int = 8


# ---------------------------------------------------------------------------
# Path + label collection
# ---------------------------------------------------------------------------

def collect_image_label_paths(
    cfg: DataConfig,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Walk cfg.image_dir, treating each subdirectory as a class.

    Returns:
        img_paths:   sorted list of absolute image file paths
        labels:      corresponding integer class indices (0 … num_classes-1)
        class_names: sorted list of class folder names (index = label)
    """
    if not os.path.isdir(cfg.image_dir):
        raise FileNotFoundError(f"image_dir not found: {cfg.image_dir}")

    class_names = sorted(
        d for d in os.listdir(cfg.image_dir)
        if os.path.isdir(os.path.join(cfg.image_dir, d))
    )

    if not class_names:
        raise FileNotFoundError(f"No class subdirectories found in {cfg.image_dir}")

    if len(class_names) != cfg.num_classes:
        raise ValueError(
            f"Expected {cfg.num_classes} class folders, found {len(class_names)}: {class_names}"
        )

    img_paths: List[str] = []
    labels:    List[int] = []

    for label_idx, cls in enumerate(class_names):
        cls_dir = os.path.join(cfg.image_dir, cls)
        files = sorted(
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(cfg.image_ext)
        )
        if not files:
            raise FileNotFoundError(
                f"No {cfg.image_ext} files found in class folder: {cls_dir}"
            )
        img_paths.extend(files)
        labels.extend([label_idx] * len(files))

    return img_paths, labels, class_names


# ---------------------------------------------------------------------------
# Stratified train / val / test split (per-class)
# ---------------------------------------------------------------------------

def split_paths(
    img_paths: List[str],
    labels:    List[int],
    cfg:       DataConfig,
    class_names: List[str],
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Stratified split: each class is split independently so the class
    distribution is preserved across train / val / test.

    Returns:
        (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)
    """
    train_imgs,  train_labels  = [], []
    val_imgs,    val_labels    = [], []
    test_imgs,   test_labels   = [], []

    for cls_idx in range(len(class_names)):
        cls_paths = [p for p, l in zip(img_paths, labels) if l == cls_idx]
        n = len(cls_paths)
        train_end = int(cfg.train_frac * n)
        val_end   = int((cfg.train_frac + cfg.val_frac) * n)

        train_imgs  += cls_paths[:train_end]
        train_labels += [cls_idx] * train_end

        val_imgs    += cls_paths[train_end:val_end]
        val_labels  += [cls_idx] * (val_end - train_end)

        test_imgs   += cls_paths[val_end:]
        test_labels += [cls_idx] * (n - val_end)

    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)


# ---------------------------------------------------------------------------
## Image loading ops (runs inside tf.data graph)
# ---------------------------------------------------------------------------

def _load_sample(
    img_path:    tf.Tensor,
    label:       tf.Tensor,
    img_size:    Tuple[int, int],
    num_classes: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    raw   = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=num_classes)
    return image, label


# ---------------------------------------------------------------------------
# Augmentation (training only — image only, no mask)
# ---------------------------------------------------------------------------

def _augment(image: tf.Tensor, label: tf.Tensor, seed: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=seed)
    return image, label


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def make_dataset(
     img_paths: List[str],
    labels:    List[int],
    cfg:       DataConfig,
    training:  bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset for one split.

    Returns:
        Batched, prefetched tf.data.Dataset yielding (image, label) pairs.
        image: (B, H, W, 3) float32 in [0, 1]
        label: (B,) int32 class indices in [0, num_classes)
    """
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices(
        (img_paths, tf.cast(labels, tf.int32))
    )

    if training:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=cfg.seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _load_sample(p, l, cfg.img_size, cfg.num_classes),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.map(
            lambda img, lbl: _augment(img, lbl, cfg.seed),
            num_parallel_calls=AUTOTUNE,
        )
    
    # For [image, image, image], label instead of image, label
    # # Convert:
    # # image -> [image, image, image]
    # ds = ds.map(
    #     lambda img, lbl: ((img, img, img), lbl),
    #     num_parallel_calls=AUTOTUNE,
    # )

    ds = ds.batch(cfg.batch_size, drop_remainder=True) # Dropping remainders to maintain batch size for SVD gradient flow
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: Optional[DataConfig] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Full pipeline: collect paths → stratified split → build train/val/test datasets.

    Usage:
        from classification_dataloader import DataConfig, build_datasets

         cfg = DataConfig(batch_size=8)
       train_ds, val_ds, test_ds, class_names = build_datasets(cfg)
        model.fit(train_ds, validation_data=val_ds, epochs=30)

    Returns:
        (train_ds, val_ds, test_ds, class_names)
        class_names[i] is the folder name for label i.
    """
    if cfg is None:
        cfg = DataConfig()

    img_paths, labels, class_names = collect_image_label_paths(cfg)

    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = split_paths(
        img_paths, labels, cfg, class_names
    )

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Dataset split — train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(test_imgs)}")

    train_ds = make_dataset(train_imgs, train_labels, cfg, training=True)
    val_ds   = make_dataset(val_imgs,   val_labels,   cfg, training=False)
    test_ds  = make_dataset(test_imgs,  test_labels,  cfg, training=False)
    return train_ds, val_ds, test_ds, class_names


# ---------------------------------------------------------------------------
# Sanity check / visualisation helper
# ---------------------------------------------------------------------------

def preview_batch(ds: tf.data.Dataset, class_names: List[str], n: int = 4) -> None:
    """Plot the first n images with their class label. Requires matplotlib."""
    import matplotlib.pyplot as plt

    images, labels = next(iter(ds))
    n = min(n, images.shape[0])

    _, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(images[i].numpy())
        axes[i].set_title(class_names[labels[i].numpy().argmax()])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()