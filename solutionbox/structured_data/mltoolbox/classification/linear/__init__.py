"""This module contains functions for multinomial logistic regression problems.

Every function can run locally or use Google Cloud Platform.
"""

from ._classification_linear import train, train_async
from mltoolbox._structured_data import (analyze, analyze_async,
                                        predict,
                                        batch_predict, batch_predict_async)
from mltoolbox._structured_data.__version__ import __version__ as __version__