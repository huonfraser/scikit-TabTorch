import pandas as pd
import numpy as np


def combineXy(X, y):
    """
    Combine X,y into a single pd.DataFrame as is expected by PyTorch Tabular
    :param X:
    :param y:
    :return:
    """

    # validate type of X
    if type(X) is np.ndarray:
        X = pd.DataFrame(X)
    elif not type(X) is pd.DataFrame():
        Exception(f"{type(X)} is unknown")

    if not y is None:
        # todo validate y
        X["Target"] = y
    return X
