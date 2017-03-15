"""This module contains functions for regression problems modeled as a fully connected feedforward deep neural network.

Every function can run locally or use Google Cloud Platform.
"""

from ._regression_dnn import train, train_async
from mltoolbox._structured_data import (analyze, analyze_async,
                                        predict,
                                        batch_predict, batch_predict_async)
from mltoolbox._structured_data.__version__ import __version__ as __version__


