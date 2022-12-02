# Data Manipulation
import numpy as np
import pandas as pd

# Data Viz
import matplotlib.pyplot as plt
import seaborn as sns

def calc_rows(cols_to_plot: list, fig_cols: int) -> int:
    return np.ceil(len(cols_to_plot) / fig_cols).astype(int)

def plot_countplot(df: pd.DataFrame, cols_to_plot: list=[], fig_cols: int=3, figsize: tuple[int]=(9.5,8), rotation=0, **kwargs):
    if len(cols_to_plot)==0:
        cols_to_plot = df.columns
    
    fig_rows = calc_rows(cols_to_plot, fig_cols)
    fig = plt.figure(figsize=figsize)
    
    for i, col in enumerate(cols_to_plot):
        ax = fig.add_subplot(fig_rows, fig_cols, i+1)
        ax = sns.countplot(
            x=col,
            data=df,
            **kwargs
        )
        plt.xticks(rotation=rotation)

def plot_violinplot(df: pd.DataFrame, cols_to_plot: list, cols_grouping: str=None, fig_cols: int=3, figsize: tuple[int]=(9.5,8), **kwargs):
    fig_rows = calc_rows(cols_to_plot, fig_cols)
    fig = plt.figure(figsize=figsize) 
    
    for i, col in enumerate(cols_to_plot):
        ax = fig.add_subplot(fig_rows, fig_cols, i+1)
        ax = sns.violinplot(
            x=col,
            y=cols_grouping,
            hue=hue,
            data=df,
            split=True,
            **kwargs
        )
    
def plot_histplot(df: pd.DataFrame, cols_to_plot: list, hue: str=None, fig_cols: int=3, figsize: tuple[int]=(9.5,8), **kwargs):
    fig_rows = calc_rows(cols_to_plot, fig_cols)
    fig = plt.figure(figsize=figsize) 
    
    for i, col in enumerate(cols_to_plot):
        ax = fig.add_subplot(fig_rows, fig_cols, i+1)
        ax = sns.histplot(
            x=col,
            data=df,
            hue=hue,
            kde=True,
            **kwargs
        )
        
def plot_intersections(y_ps_score, labels, alpha, figsize=(9.5,8)):
    from upsetplot import plot as setplot
    
    for k, a in enumerate(alpha):
        df = pd.DataFrame(y_ps_score[:,:,k], columns=labels)
        df['total'] = 1
        
        fig = plt.figure(figsize=figsize)
        setplot(df.groupby(by=list(labels)).count()['total'], fig=fig, element_size=None)