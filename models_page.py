
# seçilen farklı makine öğrenmesi algoritma modeline göre çıktı veren sayfa

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score


def show_models_page():

    df = pd.read_csv('diabetes.csv')

    X = df.iloc[:, 0:8].values
    y = df.iloc[:, -1].values

    classifier_name = st.selectbox(
        'Bir makine öğrenmesi algoritması seç',
        (#'Lineer Regresyon',
         'Lojistik Regresyon',
         'Karar Ağaçları',
         'Naif Bayes',
         'Yapay Sinir Ağları')
    )

    def add_parameter_ui(clf_name):
        params = dict()
        #if clf_name == 'Lineer Regresyon':
            #n_jobs = st.slider('n_jobs', -2, +2)
            #params['n_jobs'] = n_jobs
        if clf_name == 'Lojistik Regresyon':
            C = st.slider('C', 0.01, 10.0)
            params['C'] = C
            penalty = st.select_slider('penalty', ('l2', 'l1'))
            params['penalty'] = penalty
        elif clf_name == 'Karar Ağaçları':
            max_depth = st.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            max_features = st.slider('max_features', 1, 4)
            params['max_features'] = max_features
            min_samples_leaf = st.slider('min_samples_leaf', 1, 4)
            params['min_samples_leaf'] = min_samples_leaf
            criterion = st.select_slider('criterion', ('gini', 'entropy'))
            params['criterion'] = criterion
        elif clf_name == 'Naif Bayes':
            var_smoothing = st.slider('var_smoothing', 1e-9, 0.1)
            params['var_smoothing'] = var_smoothing
        else:
            max_iter = st.slider('max_iter', 100, 500)
            params['max_iter'] = max_iter
        return params

    st.subheader('Modelin Hiperparametreleri')
    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        clf = None
        #if clf_name == 'Lineer Regresyon':
            #clf = LinearRegression(n_jobs=params['n_jobs'])
        if clf_name == 'Lojistik Regresyon':
            clf = LogisticRegression(C=params['C'], penalty=params['penalty'])
        elif clf_name == 'Karar Ağaçları':
            clf = DecisionTreeClassifier(max_depth=params['max_depth'], max_features=params['max_features'],
                                         min_samples_leaf=params['min_samples_leaf'], criterion=params['criterion'])
        elif clf_name == 'Naif Bayes':
            clf = GaussianNB(var_smoothing=params['var_smoothing'])
        else:
            clf = MLPClassifier(max_iter=params['max_iter'], random_state=23)
        return clf

    clf = get_classifier(classifier_name, params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = clf.score(X_test, y_test)

    st.subheader('Sonuçlar')
    st.write("accuracy: ", accuracy.round(2))
    st.write("precision: ", precision_score(y_test, y_pred).round(2))
    st.write("recall: ", recall_score(y_test, y_pred).round(2))
    st.write("f1-score: ", f1_score(y_test, y_pred).round(2))

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(clf, X_test, y_test)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(clf, X_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(clf, X_test, y_test)
            st.pyplot()

    metrics = st.selectbox("Çizdirmek istediğin metriği seç",
                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    st.set_option('deprecation.showPyplotGlobalUse', False)  # hata görmemek için

    plot_metrics(metrics)

    return None