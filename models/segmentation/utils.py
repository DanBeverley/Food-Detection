import tensorflow as tf
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TqdmProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        logger.info(f"Epoch {self.current_epoch}")
        self.progress_bar = tqdm(total=self.params['steps'], desc=f"Epoch {self.current_epoch}")

    def on_batch_end(self, batch, logs=None):
        if self.progress_bar:
            self.progress_bar.update(1)
            if logs:
                desc = f"Epoch {self.current_epoch}"
                
                metrics_str = " - ".join([
                    f"{k}: {v:.4f}" for k, v in logs.items() if k not in ['batch', 'size']
                ])
                
                if metrics_str:
                    desc += f" - {metrics_str}"

                self.progress_bar.set_description(desc)

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

class StableIoU(tf.keras.metrics.Metric):
    def __init__(self, from_logits=True, name='stable_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        y_pred = tf.cast(y_pred > 0.5, self.dtype)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return self.intersection / (self.union + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)