import json
import tensorflow as tf
from typing import Tuple, Dict, Optional, Any, List
import logging
from pathlib import Path
import albumentations as A

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent

def _get_preprocess_fn():
    """Get the preprocessing function for EfficientNetV2."""
    return tf.keras.applications.efficientnet_v2.preprocess_input

def build_albumentations_pipeline(aug_cfg: Dict[str, Any]) -> A.Compose:
    """Builds an Albumentations composition for image augmentation."""
    aug_list = []
    
    if aug_cfg.get('horizontal_flip', False):
        aug_list.append(A.HorizontalFlip(p=0.5))
        logger.info("Augmentation enabled: HorizontalFlip")

    if aug_cfg.get('rotation_range', 0) > 0:
        limit = aug_cfg['rotation_range']
        aug_list.append(A.SafeRotate(limit=limit, p=0.5))
        logger.info(f"Augmentation enabled: SafeRotate (limit={limit})")

    if aug_cfg.get('shift_scale_rotate', False):
        aug_list.append(A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5))
        logger.info("Augmentation enabled: ShiftScaleRotate")

    if aug_cfg.get('coarse_dropout', False):
        aug_list.append(A.CoarseDropout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5))
        logger.info("Augmentation enabled: CoarseDropout")
        
    if aug_cfg.get('color_jitter', False):
        aug_list.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5))
        logger.info("Augmentation enabled: ColorJitter")

    if not aug_list:
        return None
        
    return A.Compose(aug_list)

@tf.function
def apply_albumentations(image, augmentation_pipeline):
    """Applies an Albumentations pipeline to a TensorFlow tensor."""
    def aug_fn(img):
        data = {"image": img.numpy()}
        aug_data = augmentation_pipeline(**data)
        aug_img = aug_data["image"]
        return aug_img

    aug_image = tf.py_function(func=aug_fn, inp=[image], Tout=tf.float32)
    aug_image.set_shape(image.get_shape())
    return aug_image

def cutmix(images, labels, alpha=1.0):
    """Apply CutMix augmentation to a batch of images and labels."""
    batch_size = tf.shape(images)[0]
    img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]
    
    lambda_val = tf.random.uniform([], 0.0, 1.0)
    lambda_val = tf.maximum(lambda_val, 1.0 - lambda_val)
    
    cut_rat = tf.sqrt(1. - lambda_val)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_rat, tf.int32)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_rat, tf.int32)
    
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    
    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
    bby2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)
    
    mask_shape = [batch_size, img_h, img_w, 1]
    mask = tf.ones(mask_shape, dtype=tf.float32)
    
    y_grid, x_grid = tf.meshgrid(tf.range(img_h), tf.range(img_w), indexing='ij')
    box_mask = tf.logical_and(
        tf.logical_and(x_grid >= bbx1, x_grid < bbx2),
        tf.logical_and(y_grid >= bby1, y_grid < bby2)
    )
    box_mask = tf.cast(box_mask, tf.float32)
    box_mask = tf.expand_dims(tf.expand_dims(box_mask, 0), -1)
    box_mask = tf.tile(box_mask, [batch_size, 1, 1, 1])
    
    mask = mask - box_mask
    
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    
    mixed_images = images * mask + shuffled_images * (1.0 - mask)
    
    box_area = tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32)
    total_area = tf.cast(img_h * img_w, tf.float32)
    lambda_adj = 1.0 - (box_area / total_area)
    
    mixed_labels = labels * lambda_adj + shuffled_labels * (1.0 - lambda_adj)
    
    return mixed_images, mixed_labels

def mixup_batch(images, labels, alpha=0.2):
    """Apply MixUp augmentation to a batch of images and labels."""
    batch_size = tf.shape(images)[0]
    
    if alpha > 0:
        lambda_mix = tf.random.uniform([], 0.0, alpha)
    else:
        lambda_mix = 1.0
    
    indices = tf.random.shuffle(tf.range(batch_size))
    
    mixed_images = lambda_mix * images + (1 - lambda_mix) * tf.gather(images, indices)
    mixed_labels = lambda_mix * labels + (1 - lambda_mix) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels

def augment_batch(images, labels, mixup_alpha=0.2, cutmix_alpha=1.0, use_cutmix_prob=0.5):
    """Apply either MixUp or CutMix augmentation randomly."""
    if tf.random.uniform([]) < use_cutmix_prob and cutmix_alpha > 0:
        return cutmix(images, labels, alpha=cutmix_alpha)
    elif mixup_alpha > 0:
        return mixup_batch(images, labels, alpha=mixup_alpha)
    else:
        return images, labels

def load_and_decode_image(path: tf.Tensor, label: tf.Tensor, image_size: tuple) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads, decodes, and resizes an image file, returning it in [0, 255] range."""
    image_data = tf.io.read_file(path)
    image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image, label

def load_classification_data(
    config: Dict[str, Any]
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], int, int, List[str], int]:
    """Loads pre-split classification dataset and applies augmentation."""
    data_cfg = config.get('data', {})
    paths_cfg = data_cfg.get('paths', {})
    
    train_metadata_path = Path(paths_cfg.get('train_metadata_file'))
    val_metadata_path = Path(paths_cfg.get('val_metadata_file'))
    label_map_path = Path(paths_cfg.get('label_map_file'))
    base_image_dir = Path(data_cfg.get('base_data_dir', '.'))

    image_size = tuple(data_cfg.get('image_size', [224, 224]))
    batch_size = data_cfg.get('batch_size', 32)
    seed = data_cfg.get('random_seed', 42)

    aug_cfg = data_cfg.get('augmentation', {})
    use_augmentation = aug_cfg.get('enabled', False)
    augmentation_pipeline = None
    if use_augmentation:
        logger.info("Building data augmentation pipeline with Albumentations...")
        augmentation_pipeline = build_albumentations_pipeline(aug_cfg)
    else:
        logger.info("Data augmentation is disabled.")

    logger.info(f"Loading train metadata from: {train_metadata_path}")
    logger.info(f"Loading val metadata from: {val_metadata_path}")
    
    if not train_metadata_path.exists() or not val_metadata_path.exists() or not label_map_path.exists():
        logger.error(f"Required files not found: train={train_metadata_path}, val={val_metadata_path}, labels={label_map_path}")
        return None, None, 0, 0, [], 0

    with open(label_map_path, 'r') as f:
        index_to_label_map = {int(k): v for k, v in json.load(f).items()}
        num_classes = len(index_to_label_map)
        class_names = [index_to_label_map[i] for i in sorted(index_to_label_map.keys())]
        label_to_index_map = {v: k for k, v in index_to_label_map.items()}

    def load_metadata_paths_labels(metadata_path: Path, dataset_name: str):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        paths, labels = [], []
        for item in metadata:
            relative_path = item.get('image_path')
            class_name = item.get('class_name')
            full_path = base_image_dir / relative_path
            if full_path.exists() and class_name in label_to_index_map:
                paths.append(str(full_path))
                labels.append(label_to_index_map[class_name])
            else:
                logger.warning(f"Skipping invalid {dataset_name} entry: path={full_path}, class={class_name}")
        
        logger.info(f"Found {len(paths)} valid {dataset_name} samples.")
        return paths, labels

    train_paths, train_labels = load_metadata_paths_labels(train_metadata_path, "training")
    val_paths, val_labels = load_metadata_paths_labels(val_metadata_path, "validation")

    if not train_paths:
        logger.error("No valid training image paths found.")
        return None, None, 0, 0, [], 0
    
    num_train_samples = len(train_paths)
    num_val_samples = len(val_paths)
    logger.info(f"Instance-based split: {num_train_samples} training, {num_val_samples} validation samples.")

    train_dataset_raw = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_dataset_raw = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)) if val_paths else None

    # Get MixUp and CutMix configuration
    mixup_alpha = aug_cfg.get('mixup_alpha', 0.0) if use_augmentation else 0.0
    cutmix_alpha = aug_cfg.get('cutmix_alpha', 0.0) if use_augmentation else 0.0
    
    if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
        logger.info(f"MixUp/CutMix augmentation enabled: mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}")

    def configure_dataset(ds: tf.data.Dataset, is_training: bool, aug_pipeline: Optional[A.Compose]) -> tf.data.Dataset:
        ds = ds.map(lambda path, label: load_and_decode_image(path, label, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            ds = ds.shuffle(1024, seed=seed)
            if aug_pipeline:
                ds = ds.map(lambda image, label: (apply_albumentations(image, aug_pipeline), label), 
                            num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.repeat()
        
        ds = ds.batch(batch_size)
        ds = ds.map(lambda img, lbl: (img, tf.one_hot(tf.cast(lbl, tf.int32), depth=num_classes)), num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training and (mixup_alpha > 0.0 or cutmix_alpha > 0.0):
            ds = ds.map(lambda img, lbl: augment_batch(img, lbl, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha), 
                       num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds
        
    train_ds = configure_dataset(train_dataset_raw, is_training=True, aug_pipeline=augmentation_pipeline)
    val_ds = configure_dataset(val_dataset_raw, is_training=False, aug_pipeline=None) if val_dataset_raw is not None else None

    return train_ds, val_ds, num_train_samples, num_val_samples, class_names, num_classes