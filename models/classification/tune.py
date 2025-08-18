import sys
import yaml
import argparse
import logging
from pathlib import Path
import os
import tensorflow as tf
import keras_tuner as kt

from data import load_classification_data
from model import build_classification_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def main(args):
    config = yaml.safe_load(Path(args.config).read_text())
    
    data_cfg = config.get('data', {})
    if args.base_data_dir:
        data_cfg['base_data_dir'] = args.base_data_dir
    config['data'] = data_cfg
    
    train_ds, val_ds, num_train, num_val, class_names, num_classes = load_classification_data(config)
    if not train_ds:
        logger.critical("Data loading failed. Cannot proceed.")
        sys.exit(1)
    logger.info(f"Data loaded: {num_train} train, {num_val} val samples.")

    model_builder = lambda hp: build_classification_model(num_classes, config, hp)

    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory='/kaggle/working/tuning_results',
        project_name='food_classification'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    logger.info("--- Starting hyperparameter search ---")
    tuner.search(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=num_train // data_cfg['batch_size'],
        validation_steps=num_val // data_cfg['batch_size'],
        callbacks=[stop_early]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n--- Search Complete ---")
    print(f"Optimal number of fine-tune layers: {best_hps.get('num_fine_tune_layers')}")
    print(f"Optimal L2 regularization: {best_hps.get('l2_regularization')}")
    print(f"Optimal dropout rate: {best_hps.get('dropout')}")
    print(f"Optimal learning rate: {best_hps.get('learning_rate')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tune a food classification model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--base_data_dir", type=str, default=None, help="Absolute path to images.")
    args = parser.parse_args()
    main(args)