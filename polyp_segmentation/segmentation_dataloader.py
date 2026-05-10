"""
Polyp segmentation data loader using tf.data.
Streams images from disk batch-by-batch — no full-dataset RAM allocation.
"""
import os
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

import tensorflow as tf


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    image_dir: str = "../Dataset/segmentation_images/images/"
    mask_dir:  str = "../Dataset/segmentation_images/masks/"
    img_size:  Tuple[int, int] = (512, 512)
    batch_size: int = 8
    train_frac: float = 0.70
    val_frac:   float = 0.20          # remainder becomes test
    seed: int = 42
    image_ext: str = ".jpg"
    mask_ext:  str = ".jpg"
    backbone: str = "densenet169"     # must match Unet(backbone_name=...)


# ---------------------------------------------------------------------------
# Path collection
# ---------------------------------------------------------------------------

def _collect_paths(directory: str, ext: str) -> list[str]:
    """Return sorted absolute paths for all files with the given extension."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(ext)
    )


def collect_image_mask_paths(
    cfg: DataConfig,
) -> Tuple[list[str], list[str]]:
    """Return (image_paths, mask_paths) guaranteed to be the same length and aligned."""
    img_paths  = _collect_paths(cfg.image_dir, cfg.image_ext)
    mask_paths = _collect_paths(cfg.mask_dir,  cfg.mask_ext)

    if len(img_paths) != len(mask_paths):
        raise ValueError(
            f"Image/mask count mismatch: {len(img_paths)} images vs {len(mask_paths)} masks"
        )
    if not img_paths:
        raise FileNotFoundError(f"No {cfg.image_ext} files found in {cfg.image_dir}")

    return img_paths, mask_paths


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_paths(
    img_paths: list[str],
    mask_paths: list[str],
    cfg: DataConfig,
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Returns three (img, mask) path tuples: (train, val, test).
    Split is deterministic — sorted input + fixed indices, no shuffle here.
    Shuffling happens inside the tf.data pipeline for the training set.
    """
    n = len(img_paths)
    train_end = int(cfg.train_frac * n)
    val_end   = int((cfg.train_frac + cfg.val_frac) * n)

    train = (img_paths[:train_end],       mask_paths[:train_end])
    val   = (img_paths[train_end:val_end], mask_paths[train_end:val_end])
    test  = (img_paths[val_end:],         mask_paths[val_end:])

    return train, val, test


# ---------------------------------------------------------------------------
# Backbone preprocessing
# ---------------------------------------------------------------------------

def get_preprocess_fn(backbone: str) -> Callable:
    """
    Return the backbone-specific preprocessing function for the given backbone.

    Uses tf.keras.applications directly for standard backbones — this avoids
    segmentation_models' internal keras.backend lookup which raises NoneType
    errors with newer TF/Keras versions. Falls back to segmentation_models only
    for non-standard backbones (e.g. resnet18) not shipped with tf.keras.
    """
    _builtin = {
        "densenet121":       tf.keras.applications.densenet.preprocess_input,
        "densenet169":       tf.keras.applications.densenet.preprocess_input,
        "densenet201":       tf.keras.applications.densenet.preprocess_input,
        "resnet50":          tf.keras.applications.resnet50.preprocess_input,
        "resnet101":         tf.keras.applications.resnet.preprocess_input,
        "resnet152":         tf.keras.applications.resnet.preprocess_input,
        "vgg16":             tf.keras.applications.vgg16.preprocess_input,
        "vgg19":             tf.keras.applications.vgg19.preprocess_input,
        "mobilenet":         tf.keras.applications.mobilenet.preprocess_input,
        "mobilenetv2":       tf.keras.applications.mobilenet_v2.preprocess_input,
        "inceptionv3":       tf.keras.applications.inception_v3.preprocess_input,
        "inceptionresnetv2": tf.keras.applications.inception_resnet_v2.preprocess_input,
    }

    if backbone in _builtin:
        return _builtin[backbone]

    # Non-standard backbone (e.g. resnet18, seresnet*) — fall back to segmentation_models
    from segmentation_models.backbones.backbones_factory import Backbones
    return Backbones.get_preprocessing(backbone)


# ---------------------------------------------------------------------------
# Image loading ops (runs inside tf.data graph)
# ---------------------------------------------------------------------------

def _decode_image(
    path: tf.Tensor,
    channels: int,
    size: Tuple[int, int]
) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=channels)
    img = tf.image.resize(img, size)        # bilinear, GPU-accelerated
    img = tf.cast(img, tf.float32)          # keep [0, 255] — preprocess_fn handles normalisation
    return img


def _load_pair(
    img_path: tf.Tensor,
    mask_path: tf.Tensor,
    img_size: Tuple[int, int],
    preprocess_fn: Callable,
) -> Tuple[tf.Tensor, tf.Tensor]:
    image = _decode_image(img_path,  channels=3, size=img_size)
    mask  = _decode_image(mask_path, channels=1, size=img_size)
    image = preprocess_fn(image)            # backbone-specific normalisation
    mask  = mask / 255.0                    # binary mask: [0, 1]
    return image, mask


# ---------------------------------------------------------------------------
# Augmentation (training only)
# ---------------------------------------------------------------------------
def _augment(image: tf.Tensor, mask: tf.Tensor, seed: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Geometric augmentations applied identically to image and mask."""
    combined = tf.concat([image, mask], axis=-1)    # (H, W, 4) — keep aligned

    combined = tf.image.random_flip_left_right(combined, seed=seed)
    combined = tf.image.random_flip_up_down(combined, seed=seed)


    image = combined[:, :, :3]
    mask  = combined[:, :, 3:]
    return image, mask


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def make_dataset(
    img_paths: list[str],
    mask_paths: list[str],
    cfg: DataConfig,
    preprocess_fn: Callable,
    training: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset for one split.

    Args:
        img_paths:    List of image file paths for this split.
        mask_paths:   Corresponding mask file paths.
        cfg:          DataConfig instance.
        preprocess_fn: Backbone preprocessing function (from get_preprocess_fn).
        training:     If True, enables shuffle + augmentation.

    Returns:
        Batched, prefetched tf.data.Dataset yielding (image, mask) pairs.
        image: (B, H, W, 3) float32, backbone-normalised
        mask:  (B, H, W, 1) float32 in [0, 1]
    """
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    if training:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=cfg.seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda img_p, mask_p: _load_pair(img_p, mask_p, cfg.img_size, preprocess_fn),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.map(
            lambda img, mask: _augment(img, mask, cfg.seed),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(cfg.batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: Optional[DataConfig] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Full pipeline: collect paths → split → build train/val/test datasets.

    Usage:
        from dataloader import DataConfig, build_datasets

        cfg = DataConfig(batch_size=8, backbone="densenet169")
        train_ds, val_ds, test_ds = build_datasets(cfg)
        model.fit(train_ds, validation_data=val_ds, epochs=50)

    Returns:
        (train_ds, val_ds, test_ds)
    """
    if cfg is None:
        cfg = DataConfig()

    img_paths, mask_paths = collect_image_mask_paths(cfg)
    (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) = split_paths(
        img_paths, mask_paths, cfg
    )

    print(f"Dataset split — train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(test_imgs)}")

    preprocess_fn = get_preprocess_fn(cfg.backbone)

    train_ds = make_dataset(train_imgs, train_masks, cfg, preprocess_fn, training=True)
    val_ds   = make_dataset(val_imgs,   val_masks,   cfg, preprocess_fn, training=False)
    test_ds  = make_dataset(test_imgs,  test_masks,  cfg, preprocess_fn, training=False)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Sanity check / visualisation helper
# ---------------------------------------------------------------------------

def preview_batch(ds: tf.data.Dataset, n: int = 3) -> None:
    """Plot the first n image-mask pairs from a dataset. Requires matplotlib."""
    import matplotlib.pyplot as plt

    images, masks = next(iter(ds))
    n = min(n, images.shape[0])

    _, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    for i in range(n):
        axes[i, 0].imshow(images[i].numpy())
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks[i].numpy().squeeze(), cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
