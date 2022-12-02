import pandas as pd

def summary(df: pd.DataFrame, complete: bool=True):
    summary = pd.concat([df.isnull().sum(), df.nunique(), df.dtypes], axis=1)
    summary.columns = ['Total Null', 'Num Unique', 'Dtype']

    print('-'*17)
    print('Data Types:')
    print(df.dtypes.value_counts().to_string())
    print('-'*17)
    print('Num Rows:', df.shape[0])
    print('Num Cols:', df.shape[1])
    
    if complete:
        print(summary)