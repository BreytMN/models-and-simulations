import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

def encoding_labels(y: pd.Series):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    return y, encoder

def train_calib_test_split(X: pd.DataFrame, y: pd.Series, split: tuple=(0.6,0.2,0.2), encoded: bool=True, random_state: int=357):
    from sklearn.model_selection import train_test_split
    
    encoder=None
    if not encoded:
        y, encoder = self.encoding_labels(y)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-split[0], random_state=357)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=split[2]/(split[1]+split[2]), random_state=357)
    
    return_list = [X_train, X_calib, X_test, y_train, y_calib, y_test]
    if not encoded:
        return_list.append(encoder)
    
    return return_list

class model_report:
    def __init__(self, X_test: pd.DataFrame, y_test: pd.Series, y_encoder=False, calibration_type: str=''):
        self.X_test = X_test
        self.y_test = y_test
        self.y_encoder = y_encoder
        self.calibration_type = calibration_type
        print(type(y_encoder))
        
    def print_classification_report(self, y_test, y_pred):
        from sklearn.metrics import classification_report
        
        print('----------------------Classification Report----------------------')
        print(classification_report(y_test, y_pred))
        
    def print_calibration_report(self, y_true, y_pred_conformal):
        accuracy = 0
        denominator = len(self.y_test)
        
        
        print('----------------------Calibration Report----------------------')
        print('test size:', denominator)
        
        for k, a in enumerate(self.alphas):
            coverage = 0
            size_0 = 0
            size_1 = 0
            size_mult = 0
            
            for i, j in enumerate(self.y_test):
                if y_pred_conformal[i, j, k]:
                    coverage += 1
                    
                if k == 0:
                    if j == y_true[i]:
                        accuracy +=1
                        
                size = sum(y_pred_conformal[i, :, k])
                if size == 0: size_0 +=1
                if size == 1: size_1 +=1
                if size >= 2: size_mult +=1
            
            if k == 0:
                print('uncalibrated accuracy:', round((accuracy / denominator)*100, 2), '%')
            print('-----------------')
            print('alpha:', a)
            print('coverage:', round((coverage / denominator)*100, 2), '%')
            print('empty sets:', size_0)
            print('unit sets:', size_1)
            print('other sets:', size_mult)
            
    def plot_confusion_matrix(self, y_test, y_pred, labels, ax):
        from sklearn.metrics import confusion_matrix
        
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            xticklabels=labels, yticklabels=labels,
            annot=True, cmap='Blues', fmt='g',
            ax=ax
        )
        ax.set_title('Confusion Matrix', fontsize=20)
        ax.set_xlabel('Predicted Labels', fontsize=14)
        ax.set_ylabel('True Labels', fontsize=14)
        
    def lightgbm_classifier(self, model):
        from lightgbm import plot_importance, plot_metric

        y_pred = model.predict(self.X_test)
        y_test = self.y_test
        labels = model.classes_
        print(labels)
        if self.y_encoder: 
            y_pred = self.y_encoder.inverse_transform(y_pred)
            y_test = self.y_encoder.inverse_transform(y_test)
            labels = self.y_encoder.inverse_transform(labels)
            print(labels)

        self.print_classification_report(y_test, y_pred)

        fig, ax = plt.subplots(1, 3, figsize=(30, 8))
        self.plot_confusion_matrix(y_test, y_pred, labels, ax[0])
        plot_metric(model, ax=ax[1])
        ax[1].set_title('Training Metrics', fontsize=20)
        ax[1].set_xlabel('Iterations', fontsize=14)
        ax[1].set_ylabel('Loss', fontsize=14)
        plot_importance(model, ax=ax[2])
        ax[2].set_title('Feature Importances', fontsize=20)
        ax[2].set_xlabel('Importances', fontsize=14)
        ax[2].set_ylabel('', fontsize=14);
        
    def calibration_classifier(self, conformal_model):
        from mapie.metrics import classification_coverage_score, classification_mean_width_score
        from matplotlib import cm
        
        y_true, y_pred_conformal = conformal_model.predict(self.X_test, self.alphas)
        self.print_calibration_report(y_true, y_pred_conformal)
        
        colors = iter(cm.Dark2(np.linspace(0, 1, len(self.alphas))))
        scores = conformal_model.conformity_scores_
        n = conformal_model.n_samples_
        quantiles = conformal_model.quantiles_
        
        fig, ax = plt.subplots(1, 3, figsize=(30, 8))
        sns.histplot(scores, bins=20, binrange=(0,1), ax=ax[0])
        for i, quantile in enumerate(quantiles):
            c = next(colors)
            ax[0].axvline(
                x = quantile,
                color=c, linestyle="dashed",
                label=f"alpha = {self.alphas[i]}"
            )
        ax[0].set_title("Distribution of scores")
        ax[0].legend()
        ax[0].set_xlabel("Scores")
        ax[0].set_ylabel("Count")
        
        alpha_range = np.arange(0.01, 1, 0.01)
        _, y_pred_conformal_range = conformal_model.predict(self.X_test, alpha=alpha_range)
        
        coverages = [classification_coverage_score(self.y_test, y_pred_conformal_range[:, :, i])
                     for i, _ in enumerate(alpha_range)]
    
        ax[1].scatter(1 - alpha_range, coverages, label='score')
        ax[1].set_xlabel("1 - alpha")
        ax[1].set_ylabel("Coverage score")
        ax[1].plot([0, 1], [0, 1], label="x=y", color="black")
        ax[1].legend()
        
        sizes = [classification_mean_width_score(y_pred_conformal_range[:, :, i])
                 for i, _ in enumerate(alpha_range)]
        
        ax[2].scatter(1 - alpha_range, sizes, label='score')
        ax[2].set_xlabel("1 - alpha")
        ax[2].set_ylabel("Average size of prediction sets")
        ax[2].legend()
        plt.show();
        
        
    def calibration_report(self, conformal_model, alphas: list=[0.2, 0.15, 0.1, 0.05]):
        self.alphas = alphas
        
        if self.calibration_type=='classifier':
            self.calibration_classifier(conformal_model)