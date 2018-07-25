import pandas as pd
import numpy as np

def types(df):
    """
        DataFrame column types.
        A column is either of 'Binary', 'Categorical' or 'Numerical' type.
    """
    types = list()
    for column in df.columns:
        # Number of unique values of a column.
        unique = len(df[column].unique())
        if unique <= 2:
            types.append('Binary')
        elif df.dtypes[column] in ['object', 'int64']:
            types.append('Categorical')
        else:
            types.append('Numerical')
    return pd.Series(types, index=df.columns, name='type')

def sub(df, type):
    df_types = types(df)
    type = type.split('_')
    if not set(type).issubset({'Numerical', 'Categorical', 'Binary'}):
        raise OSError('Type not in the good format. Please use _ between types.')
    selection = df[df_types[df_types.isin(type)].index]
    return selection