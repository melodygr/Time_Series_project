import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import user_functions
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, plot_confusion_matrix, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder, StandardScaler, scale
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectKBest, RFE, RFECV
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


def print_roc(false_positive_rate, true_positive_rate):    
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print('AUC: {}'.format(auc(false_positive_rate, true_positive_rate)))
    print('----------------------------------------------')
    plt.plot(false_positive_rate, true_positive_rate, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/10.0 for i in range(11)])
    plt.xticks([i/10.0 for i in range(11)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
def print_confusion_matrices(model, X_train, X_test, y_train, y_test, y_train_preds, y_test_preds):
    print('\nTRAIN Confusion Matrix')
    print('----------------')
    plot_confusion_matrix(model, X_train, y_train, values_format='.8g')
    print("Number of mislabeled training points out of a total {} points : {}, percentage = {:.4%}"
          .format(X_train.shape[0], (y_train != y_train_preds).sum(), (y_train != y_train_preds).sum()/X_train.shape[0]))
    plt.show()
    print('\nTEST Confusion Matrix')
    print('----------------')
    plot_confusion_matrix(model, X_test, y_test, values_format='.4g')
    print("Number of mislabeled test points out of a total {} points : {}, percentage = {:.4%}"
          .format(X_test.shape[0], (y_test != y_test_preds).sum(), (y_test != y_test_preds).sum()/X_test.shape[0]))
    plt.savefig('Images/confusion_matrix.png')
    plt.show()
    
def print_metrics(model, X_train, X_test, y_train, y_test, y_train_preds, y_test_preds):
    
    # Calculate scores
    train_precision = precision_score(y_train, y_train_preds)
    test_precision = precision_score(y_test, y_test_preds)
    train_recall = recall_score(y_train, y_train_preds)
    test_recall = recall_score(y_test, y_test_preds)
    train_accuracy = accuracy_score(y_train, y_train_preds)
    test_accuracy = accuracy_score(y_test, y_test_preds)
    train_f1 = f1_score(y_train, y_train_preds)
    test_f1 = f1_score(y_test, y_test_preds)
    
    # Print scores
    print("Precision Score: Train {0:.5f}, Test {1:.5f}".format(train_precision, test_precision))
    print("Recall Score:\t Train {0:.5f}, Test {1:.5f}".format(train_recall, test_recall))
    print("Accuracy Score:\t Train {0:.5f}, Test {1:.5f}".format(train_accuracy, test_accuracy))
    print("F1 Score: \t Train {0:.5f}, Test {1:.5f}".format(train_f1, test_f1))
    print('----------------')
    
    # Create and print train & test confusion matrices 
    print_confusion_matrices(model, X_train, X_test, y_train, y_test, y_train_preds, y_test_preds)
    print('----------------')  
    
    # print classification report
    print(classification_report(y_test, y_test_preds))
    
    # Check the AUC for predictions
    if str(model)[:3] == 'Log':
        y_score = model.fit(X_train, y_train).decision_function(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_score)
        print_roc(false_positive_rate, true_positive_rate)
    
    if str(model)[:3] == 'Dec':
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_test_preds)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print('\nAUC is :{0}'.format(round(roc_auc, 2)))
        fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (12,12), dpi=500)
        tree.plot_tree(model, feature_names = X_train.columns, 
               class_names=np.unique(y_test).astype('str'),
               filled = True)
        plt.show()
        
    return train_precision, test_precision, train_recall, test_recall, train_accuracy, test_accuracy, train_f1, test_f1    
    
def run_model(model, name, X_train, X_test, y_train, y_test, metrics_df, fit_X = None, fit_y = None):
    
    if fit_X.empty:
        fit_X = X_train
        fit_y = y_train
    
    model_metrics = [name, model]
    
    tic = time.time()
    model.fit(fit_X, fit_y)
    traintime = time.time() - tic
    print('Fit time: ', traintime)
    model_metrics.append(traintime)
    
    # Calculate train and test predictions
    toc = time.time()
    y_test_preds = model.predict(X_test)
    y_train_preds = model.predict(X_train)
    predtime = time.time() - toc
    print('Prediction time: ', predtime)
    model_metrics.append(predtime)
    
    train_precision, test_precision, train_recall, test_recall, train_accuracy, test_accuracy, train_f1, test_f1 = print_metrics(model, X_train, X_test, y_train, y_test, y_train_preds, y_test_preds)
    
    model_metrics.extend([train_precision, test_precision, train_recall, test_recall, train_accuracy, test_accuracy, train_f1, test_f1])
    
    series = pd.Series(model_metrics, index = metrics_df.columns)
    metrics_df = metrics_df.append(series, ignore_index=True)
    
    return metrics_df

def plot_ten_feature_importances(model, X_train, name):

    ten_features = sorted(list(zip(model.feature_importances_, X_train.columns.values)), reverse=True)[:10]
    importances, labels = map(list,zip(*reversed(ten_features)))
    plt.figure(figsize=(8,8))
    plt.barh(range(10), importances, align='center') 
    plt.yticks(np.arange(10), labels) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    file_name = 'Images/feature_importance' + name + '.png'
    print(file_name)
    plt.savefig(file_name)
    
def plot_feature_importances(model, X_train):
    
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    