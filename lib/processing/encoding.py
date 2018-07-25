import utilities

import numpy as np
import pandas as pd
import collections

from sklearn.decomposition import PCA

def likelihood(df, type='Categorical'):
    """ 
        Performs likelihood encoding.
            
        :param df: Data
        :param column: Column to encode
        :return: Encoded data
        :rtype: pd.Dataframe
    """
    # Columns' type.
    x = df.copy()

    selection = utilities.sub(x, type=type)
    numericals = utilities.sub(x, type='Numerical')

    # PCA.
    pca = PCA()
    # First principal axe.
    principal_axe = pca.fit(numericals.values).components_[0, :]
    # First principal component.
    principal_component = (principal_axe * numericals).sum(axis=1)

    for column in selection.columns:
        categories = x[column].unique()
        for category in categories:
            selection.loc[selection[column] == category, column] = np.mean(principal_component[x[column]==category])

    x.update(selection)

    return x

def label(df, type='binary'):
    """ 
        Performs likelihood encoding.
            
        :param df: Data
        :param type: Data type to encode
        :return: Encoded data
        :rtype: pd.DataFrame
    """
    x = df.copy()

    selection = utilities.sub(x, type=type)

    for column in selection.columns:
        if x[column].dtype != np.int64:
            categories = x[column].unique()
            for i, category in enumerate(categories):
                    selection.loc[selection[column] == category, column] = i
    x.update(selection)

    return x

def one_hot(df, encoding=None, type='All'):
    """
        Performs one-hot encoding.

        :param df: Data
        :param type: Data type to encode
        :return: Encoded data
        :rtype: pd.DataFrame
    """
    x = label(df, type=type)
    selection = utilities.sub(x, type=type)
    for column in selection.columns:
        x = pd.concat([x, pd.get_dummies(x[column], prefix=column)],axis=1)
        x.drop([column],axis=1, inplace=True)
    return x
