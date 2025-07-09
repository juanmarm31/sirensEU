"""
Created on 08.01.2019 

@author: goh
"""

from tensorflow.keras import optimizers
from ..logging_alf import getLogger
import math

from abc import ABC, abstractmethod


LOGGER = getLogger()


def getOptimizer(**optimizeParams):
    optimizer = getattr(optimizers, optimizeParams["optimizer"]) if "optimizer" in optimizeParams else getattr(optimizers, "Adam")
    parameters = optimizeParams["parameters"] if "parameters" in optimizeParams else {}
    return optimizer(**parameters)


class AbstractLrScheduler(ABC):
    def __init__(self, **params):
        """ Additional params that can be set via config. """
        super().__init__()

    @abstractmethod
    def custom_learning_rate_function(self, epoch, lr):
        """ Interface that will be used as keras callback. """
        pass


class CosineWRScheduler(AbstractLrScheduler):
    def __init__(self, **params):
        """ Learning rate scheduler using cosine learning rate decay. Optional with warm restarts that resets learning rate after first_restart_interval.

            Valid parameters are:
                epochs: required! must be same as in config
                learning_rate: initial learning rate
                minimum_lr: minimum learning rate
                first_restart_interval: when to reset learning rate
                double_restart_interval: if true interval for restarts will be doubled each time.
                    Please check that it fits to number of epochs at the end. Otherwise interval maybe very short.
            Args:
                params (**kwargs): dict with additional parameters (alf interface)
            """
        self.epochs = params.get("epochs")
        self.learning_rate = params.get("learning_rate", 1e-3)
        self.minimum_lr = params.get("minimum_lr", 1e-6)
        # when to reset learning rate
        self.restart_int = params.get("first_restart_interval", self.epochs)
        # if true interval for restarts will be doubled each time. Please check that it fits to number of epochs at the end.
        self.double_restart_interval = params.get("double_restart_interval", True)
        self.epoch_since_restart = 0

    def custom_learning_rate_function(self, epoch, lr):
        """ Keras learning rate callback """
        if self.epoch_since_restart == self.restart_int:
            if self.double_restart_interval:
                self.restart_int *= 2
            self.epoch_since_restart = 0
            # last interval cool fully down in remaining epochs
            if self.epochs - epoch < self.restart_int:
                self.restart_int = self.epochs - epoch
        else:
            self.epoch_since_restart += 1
        lr = self.minimum_lr + 0.5*(self.learning_rate-self.minimum_lr)*(1+math.cos(self.epoch_since_restart/self.restart_int*math.pi))
        return lr
