import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

def encoding_labels(y: pd.Series):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    return y, encoder

def report(X_test: pd.DataFrame, y_test: pd.Series, model, y_encoder):
    from sklearn.metrics import classification_report, confusion_matrix
    from lightgbm import plot_importance, plot_metric
    
    y_pred = y_encoder.inverse_transform(model.predict(X_test))
    y_test = y_encoder.inverse_transform(y_test)
    labels = y_encoder.inverse_transform(model.classes_)
    
    print('----------------------Classification Report----------------------')
    print(classification_report(y_test, y_pred))
    
    fig, ax = plt.subplots(1, 3, figsize=(30, 8))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        xticklabels=labels, yticklabels=labels,
        annot=True, cmap='Blues', fmt='g',
        ax=ax[0]
    )
    
    plot_metric(model, ax=ax[1])
    plot_importance(model, ax=ax[2])
    
    ax[0].set_title('Confusion Matrix', fontsize=20)
    ax[0].set_xlabel('Predicted Labels', fontsize=14)
    ax[0].set_ylabel('True Labels', fontsize=14)
    
    ax[1].set_title('Training Metrics', fontsize=20)
    ax[1].set_xlabel('Iterations', fontsize=14)
    ax[1].set_ylabel('Loss', fontsize=14)
    
    ax[2].set_title('Feature Importances', fontsize=20)
    ax[2].set_xlabel('Importances', fontsize=14)
    ax[2].set_ylabel('', fontsize=14);
    
def calibration_report(X_test, y_test, conformal_model, alphas):
    from matplotlib import cm
    colors = iter(cm.Dark2(np.linspace(0, 1, len(alphas))))
    
    scores = conformal_model.conformity_scores_
    n = conformal_model.n_samples_
    quantiles = conformal_model.quantiles_
    
    fig, ax = plt.subplots(1, 3, figsize=(30, 8))
    
    sns.histplot(scores, bins=20, binrange=(0,1), ax=ax[0])
    for i, quantile in enumerate(quantiles):
        c = next(colors)
        ax[0].axvline(
            x = quantile,
            #ymin=0, ymax=400,
            color=c, linestyle="dashed",
            label=f"alpha = {alphas[i]}",
            #ax=ax[0]
        )
    ax[0].set_title("Distribution of scores")
    ax[0].legend()
    plt.xlabel("Scores")
    plt.ylabel("Count")
    
    from mapie.metrics import classification_coverage_score, classification_mean_width_score
    
    alpha_range = np.arange(0.01, 1, 0.01)
    _, y_pred_conformal_range = conformal_model.predict(X_test, alpha=alpha_range)
    
    coverages = [classification_coverage_score(y_test, y_pred_conformal_range[:, :, i])
                 for i, _ in enumerate(alpha_range)]
    sizes = [classification_mean_width_score(y_pred_conformal_range[:, :, i])
             for i, _ in enumerate(alpha_range)]

    ax[1].scatter(1 - alpha_range, coverages, label='score')
    ax[1].set_xlabel("1 - alpha")
    ax[1].set_ylabel("Coverage score")
    ax[1].plot([0, 1], [0, 1], label="x=y", color="black")
    ax[1].legend()
    
    ax[2].scatter(1 - alpha_range, sizes, label='score')
    ax[2].set_xlabel("1 - alpha")
    ax[2].set_ylabel("Average size of prediction sets")
    ax[2].legend()
    plt.show();

def train_calib_test_split(X: pd.DataFrame, y: pd.Series, split=(0.6,0.2,0.2), random_state=357):
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-split[0], random_state=357)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=split[2]/(split[1]+split[2]), random_state=357)
    
    return X_train, X_calib, X_test, y_train, y_calib, y_test