# -*- coding: utf-8 -*-
"""
Callbacks

This module contains the Callbacks class which provides methods to create various Keras callbacks for training.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback


class TrainingMonitor(Callback):
    """ Training Monitor to monitor training at each epoch
    """
    def __init__(self, fig_path: str, json_path: Optional[str]=None, startAt: int=0):
        """ Initialize the TrainingMonitor

            Args:
                fig_path(str): Path to save training plots
                json_path(Optional[str]): Path to save training results as json
                startAt(int): Starting epoch
        """
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor,self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.startAt = startAt

    def on_train_begin(self, logs: Optional[dict]=None):
        """ Initialize the history dictionary

            Args:
                logs(Optional[dict]): Training logs
        """
        logs = logs or {}
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch: int, logs: dict={}):
        """ Update the history dictionary and plot the training history

            Args:
                epoch(int): Current epoch
                logs(dict): Training logs
        """
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            plt.savefig(self.fig_path)
            plt.close()


class Callbacks:
    """ Callbacks for training """
    def __init__(self, output_path: str):
        """ Initialize the output path

            Args:
                output_path(str): Path to save callback outputs
        """
        self._output_path = output_path

    def get_callbacks(self, reduce_lr_on_plateau: bool=True, early_stopping: bool=True, model_checkpoint: bool=True, tensorboard: bool=True, training_monitor: bool=True):
        """ Get the list of callbacks

            Args:
                reduce_lr_on_plateau(bool): Whether to use ReduceLROnPlateau callback
                early_stopping(bool): Whether to use EarlyStopping callback
                model_checkpoint(bool): Whether to use ModelCheckpoint callback
                tensorboard(bool): Whether to use TensorBoard callback
                training_monitor(bool): Whether to use TrainingMonitor callback

            Returns:
                list: List of callbacks
        """
        callbacks = []

        # add model checkpoint
        if model_checkpoint:
            ckpoint_path = os.path.join(self._output_path, "ck_pt")
            if not os.path.exists(ckpoint_path):
                os.makedirs(ckpoint_path)
            callbacks.append(Callbacks.model_checkpoint(ckpoint_path))
        
        # add reduce learning rate on plateau
        if reduce_lr_on_plateau:
            callbacks.append(Callbacks.reduce_lr_on_plateau())
        
        # add early stopping
        if early_stopping:
            callbacks.append(Callbacks.early_stopping())

        # add tensorboard
        if tensorboard:
            tensorboard_path = os.path.join(self._output_path, "tensorboard")
            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path)
            callbacks.append(Callbacks.tensorboard(tensorboard_path))

        # add training monitor
        if training_monitor:
            fig_path = os.path.sep.join([self._output_path, "output", "{}.png".format(os.getpid())])
            json_path = os.path.sep.join([self._output_path, "json", "{}.json".format(os.getpid())])

            if not os.path.exists(os.path.dirname(fig_path)):
                os.makedirs(os.path.dirname(fig_path))
            if not os.path.exists(os.path.dirname(json_path)):
                os.makedirs(os.path.dirname(json_path))
            callbacks.append(Callbacks.training_monitor(fig_path, json_path))

        return callbacks

    @staticmethod
    def training_monitor(fig_path: str, json_path: str):
        """ Monitor training at each epoch

            Args:
                fig_path(str): Path to save training plots
                json_path(str): Path to save training results as json

            Returns:
                TrainingMonitor: TrainingMonitor callback
        """
        return TrainingMonitor(fig_path, json_path)

    @staticmethod
    def reduce_lr_on_plateau(metric_to_monitor: str='val_loss', mode: str="min", factor: float=0.2, patience: int=5, min_lr: float=0.001, min_delta: float=0.001):
        """ Callback to reduce learning rate on plateau

            KwArgs:
                metric_to_monitor(str): Quantity to be monitored.
                mode(str): When to stop training in mode(min, max, auto)
                factor(float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
                patience(int): Number of epochs with no improvement after which learning rate will be reduced.
                min_lr(float): Lower bound on the learning rate.
                min_delta(float): Threshold for measuring the new optimum, to only focus on significant changes.

            Returns:
                ReduceLROnPlateau: ReduceLROnPlateau callback
        """
        return ReduceLROnPlateau(
            monitor=metric_to_monitor, factor=factor, patience=patience, min_lr=min_lr,
            min_delta=min_delta, mode=mode, verbose=1
        )

    @staticmethod
    def early_stopping(monitor: str='val_loss', mode: str='min', patience: int=50, restore_best_weights: bool=True):
        """ Stop training when a monitored metric has stopped improving.

            KwArgs:
                monitor(str): Quantity to be monitored.
                mode(str): When to stop training in mode(min, max, auto)
                patience(int): Number of epochs with no improvement after which training will be stopped
                restore_best_weights(bool): Whether to restore model weights from the epoch with the best value of the monitored quantity

            Returns:
                EarlyStopping: EarlyStopping callback
        """
        return EarlyStopping(monitor=monitor, mode=mode, verbose=1, patience=patience, restore_best_weights=restore_best_weights)

    @staticmethod
    def model_checkpoint(dirname: str, monitor: str="val_loss", mode: str="min", save_best_only: bool=True):
        """ Save only the *best* model to disk based on the validation loss

            Args:
                dirname(str): Name of the directory to save model

            KwArgs:
                monitor(str): Quantity to be monitored
                mode(str): When to stop training in mode(min, max, auto)
                save_best_only(bool): Restore model weights from the epoch with the best value of the monitored quantity

            Returns:
                ModelCheckpoint: ModelCheckpoint callback
        """
        fname = os.path.join(dirname, "weights-{epoch:03d}-{val_loss:.4f}.keras")
        return ModelCheckpoint(fname, monitor=monitor, mode=mode, save_best_only=save_best_only, verbose=1)

    @staticmethod
    def tensorboard(log_dir: str="./logs", profile_batch: int=5):
        """ Enable visualizations for TensorBoard.

            KwArgs:
                log_dir(str): Path of the directory to save the log files
                profile_batch(int): Profile the batch

            Returns:
                TensorBoard: TensorBoard callback
        """
        return TensorBoard(log_dir=log_dir, profile_batch=profile_batch)