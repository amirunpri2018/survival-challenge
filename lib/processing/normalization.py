import pandas as pd
import numpy as np

import utilities

def standard(df, type='Numerical'):
    """ 
        Performs standard normalization.
            
        :param df: Data
        :param type: Data type to normalize
        :return: Normalized data
        :rtype: pd.DataFrame
    """
    x = df.copy()
    selection = utilities.sub(x, type=type)

    for column in selection.columns:
        selection[column] = (selection[column] - selection[column].mean()) / selection[column].std()

    x.update(selection)

    return x
