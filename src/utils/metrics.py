import pandas as pd 
import numpy as np 
from src.modeling.model_pipline import ModelPipline 
from src.modeling.model_xgb import XGBoost 
import xgboost as xgb 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle 
import warnings 
warnings.filterwarnings("ignore")

# roc_score and confustion_matrix scores 
def evaluate_sk(model, x_train, x_test, y_train, y_test, flg=True):
    """sklearn models"""
    pred_tr = model.predict_proba(x_train)[:, 1].ravel()
    pred_te = model.predict_proba(x_test)[:, 1].ravel()
    
    pred_tr_ = model.predict(x_train).ravel()
    pred_te_ = model.predict(x_test).ravel()
    
    true_tr = y_train.values.ravel()
    true_te = y_test.values.ravel()
    
    tr_f, tr_t, tr_th = roc_curve(true_tr, pred_tr)
    te_f, te_t, te_th = roc_curve(true_te, pred_te)
    if flg:
        roc_tr = roc_auc_score(pred_tr_, true_tr)
        roc_te = roc_auc_score(pred_te_, true_te)
    plt.subplot(2, 2, 1)
    plt.plot(tr_f, tr_t, marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if flg:
        plt.title(f"Train: {roc_tr:.4f}")
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(te_f, te_t, marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if flg:
        plt.title(f"Val: {roc_te:.4f}")
    plt.grid()
    
    conf = confusion_matrix(pred_tr_, true_tr)
    plt.subplot(2, 2, 3)
    sns.heatmap(conf, annot=True, fmt="", cmap="Blues")
    plt.xlabel("correct")
    plt.ylabel("predict")
    
    conf = confusion_matrix(pred_te_, true_te)
    plt.subplot(2, 2, 4)
    sns.heatmap(conf, annot=True, fmt="", cmap="Blues")
    plt.tight_layout()
    
    
def evaluate(model, x_train, x_test, y_train, y_test):
    """xgboost models"""
    pred_tr = model.predict(xgb.DMatrix(x_train))
    pred_te = model.predict(xgb.DMatrix(x_test))
    
    true_tr = y_train.values.ravel()
    true_te = y_test.values.ravel()
    
    tr_f, tr_t, tr_th = roc_curve(true_tr, pred_tr)
    te_f, te_t, te_th = roc_curve(true_te, pred_te)
    roc_tr = roc_auc_score(np.where(pred_tr >= .5, 1, 0), true_tr)
    roc_te = roc_auc_score(np.where(pred_te >= .5, 1, 0), true_te)
    plt.subplot(2, 2, 1)
    plt.plot(tr_f, tr_t, marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Train: {roc_tr:.4f}")
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(te_f, te_t, marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Val: {roc_te:.4f}")
    plt.grid()
    
    conf = confusion_matrix(np.where(pred_tr >= .5, 1, 0), true_tr)
    plt.subplot(2, 2, 3)
    sns.heatmap(conf, annot=True, fmt="", cmap="Blues")
    plt.xlabel("correct")
    plt.ylabel("predict")
    
    conf = confusion_matrix(np.where(pred_te >= .5, 1, 0), true_te)
    plt.subplot(2, 2, 4)
    sns.heatmap(conf, annot=True, fmt="", cmap="Blues")
    plt.tight_layout()