"""
This module defines custom callbacks for PyTorch Lightning training.
It includes callbacks for logging metrics, saving model checkpoints, early stopping, and monitoring learning rates.
"""
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import matplotlib.pyplot as plt


class LogMetricsCallback(pl.Callback):
    """
    Callback to log training metrics and save them as PNG images.
    """
    def __init__(self, output_path):
        self.output_path = output_path

    def on_epoch_end(self, trainer, pl_module):
        """
        Called when the epoch ends. Logs the training and validation loss and accuracy, and saves the plots as PNG images.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The PyTorch Lightning module.
        """
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Plot the training loss and accuracy
        plt.figure()
        plt.plot(metrics['train_loss'], label='train_loss')
        plt.plot(metrics['val_loss'], label='val_loss')
        plt.plot(metrics['train_acc'], label='train_acc')
        plt.plot(metrics['val_acc'], label='val_acc')
        plt.title(f"Training Loss and Accuracy [Epoch {epoch}]")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        # Save the figure
        plt.savefig(os.path.join(self.output_path, f"metrics_epoch_{epoch}.png"))
        plt.close()

class Callbacks:
    """
    Class to manage and provide various callbacks for PyTorch Lightning training.
    """
    def __init__(self, output_path: str="."):
        """ Initialize the output path """
        self._output_path = output_path

    def get_callbacks(self, learning_rate_monitor: bool=True, log_metrics: bool=True, model_checkpoint: bool=True, early_stopping: bool=True):
        """
        Get the list of callbacks based on the provided flags.

        Args:
            learning_rate_monitor (bool): Whether to include the learning rate monitor callback.
            log_metrics (bool): Whether to include the log metrics callback.
            model_checkpoint (bool): Whether to include the model checkpoint callback.
            early_stopping (bool): Whether to include the early stopping callback.

        Returns:
            list: List of callbacks.
        """
        callbacks = []

        if model_checkpoint:
            ckpoint_path = os.path.join(self._output_path, "ck_pt")
            if not os.path.exists(ckpoint_path):
                os.makedirs(ckpoint_path)
            callbacks.append(Callbacks.model_checkpoint(ckpoint_path))

        if early_stopping:
            callbacks.append(Callbacks.early_stopping())

        if learning_rate_monitor:
            callbacks.append(Callbacks.learning_rate_monitor())
        
        if log_metrics:
            log_metrics_path = os.path.join(self._output_path, "metrics")
            callbacks.append(Callbacks.log_metrics(output_path=log_metrics_path))

        return callbacks

    @staticmethod
    def model_checkpoint(dirname: str, monitor: str="val_loss", mode: str="min"):
        """
        Save only the best model to disk based on the validation loss.

        Args:
            dirname (str): Name of the directory to save the model.
            monitor (str): Quantity to be monitored.
            mode (str): When to stop training in mode (min, max, auto).

        Returns:
            ModelCheckpoint: The model checkpoint callback.
        """
        return ModelCheckpoint(
            monitor=monitor,
            dirpath=dirname,
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode=mode,
        )

    @staticmethod
    def early_stopping(monitor: str='val_loss', mode: str='min', patience: int=5, restore_best_weights: bool=True):
        """
        Stop training when a monitored metric has stopped improving.

        Args:
            monitor (str): Quantity to be monitored.
            mode (str): When to stop training in mode (min, max, auto).
            patience (int): Number of epochs with no improvement after which training will be stopped.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value of the monitored quantity.

        Returns:
            EarlyStopping: The early stopping callback.
        """
        return EarlyStopping(
            monitor=monitor,
            patience=patience,
            verbose=True,
            mode=mode
        )
        
    @staticmethod
    def learning_rate_monitor(logging_interval: str="epoch"):
        """
        Monitor the learning rate during training.

        Args:
            logging_interval (str): Interval for logging the learning rate (epoch or step).

        Returns:
            LearningRateMonitor: The learning rate monitor callback.
        """
        return LearningRateMonitor(logging_interval=logging_interval)
    
    @staticmethod
    def log_metrics(output_path: str='metrics_logs'):
        """
        Log training metrics and save them as PNG images.

        Args:
            output_path (str): Path to save the metrics logs.

        Returns:
            LogMetricsCallback: The log metrics callback.
        """
        return LogMetricsCallback(output_path=output_path)

