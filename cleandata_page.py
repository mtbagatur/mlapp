
# anlamlı hale getirilmiş veri setinin görselleştirilmesi

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score

def show_cleandata_page():
    df = pd.read_csv("diabetes.csv")

    df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = df[["Glucose", "BloodPressure", "SkinThickness",
                                                                              "Insulin", "BMI"]].replace(0, np.NaN)

    naValues = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for i in naValues:
        df[i][(df[i].isnull()) & (df["Outcome"] == 0)] = df[i][(df[i].isnull()) & (df["Outcome"] == 0)].fillna(
            df[i][df["Outcome"] == 0].mean())
        df[i][(df[i].isnull()) & (df["Outcome"] == 1)] = df[i][(df[i].isnull()) & (df["Outcome"] == 1)].fillna(
            df[i][df["Outcome"] == 1].mean())

    for feature in df:

        Q1 = df[feature].quantile(0.05)
        Q3 = df[feature].quantile(0.95)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR

        if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):
            print(feature, "yes")
            print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
            print("lower", lower, "\nupper", upper)
            df.loc[df[feature] > upper, feature] = upper
        else:
            print(feature, "no")

        df['BMIRanges'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=["Underweight", "Healthy",
                                                                                       "Overweight", "Obese"])

        def set_insulin(row):
            if row["Insulin"] >= 16 and row["Insulin"] <= 166:
                return "Normal"
            else:
                return "Abnormal"

        df = df.assign(INSULIN_DESC=df.apply(set_insulin, axis=1))

        df['NewGlucose'] = pd.cut(x=df['Glucose'], bins=[0, 70, 99, 126, 200],
                                  labels=["Low", "Normal", "Secret", "High"])

        df = pd.get_dummies(df, drop_first=True)

        r_scaler = RobustScaler()
        df_r = r_scaler.fit_transform(df.drop(["Outcome", "BMIRanges_Healthy", "BMIRanges_Overweight",
                                               "BMIRanges_Obese", "INSULIN_DESC_Normal", "NewGlucose_Normal",
                                               "NewGlucose_Secret", "NewGlucose_High"], axis=1))

        df_r = pd.DataFrame(df_r, columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

        df = pd.concat([df_r, df[["Outcome", "BMIRanges_Healthy", "BMIRanges_Overweight",
                                  "BMIRanges_Obese", "INSULIN_DESC_Normal", "NewGlucose_Normal",
                                  "NewGlucose_Secret", "NewGlucose_High"]]], axis=1)

        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

        classifier_name = st.selectbox(
            'Bir makine öğrenmesi algoritması seç',
            (  # 'Lineer Regresyon',
                'Lojistik Regresyon',
                'Karar Ağaçları',
                'Naif Bayes',
                'Yapay Sinir Ağları')
        )

        def add_parameter_ui(clf_name):
            params = dict()
            # if clf_name == 'Lineer Regresyon':
            # n_jobs = st.slider('n_jobs', -2, +2)
            # params['n_jobs'] = n_jobs
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
            # if clf_name == 'Lineer Regresyon':
            # clf = LinearRegression(n_jobs=params['n_jobs'])
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