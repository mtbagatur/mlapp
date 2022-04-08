
# kullanıcı girdilerine göre tahmin sayfası

import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def show_predict_page():

    df = pd.read_csv("diabetes.csv")

    X = df.iloc[:, 0:8].values
    y = df.iloc[:, -1].values

    def get_user_input():
        pregnancies = st.slider('hamilelik', 0, 17, 3)
        glucose = st.slider('glikoz', 0, 199, 117)
        blood_pressure = st.slider('kan basıncı', 0, 122, 72)
        skin_thickness = st.slider('deri kalınlığı', 0, 99, 23)
        insulin = st.slider('insülin', 0.0, 846.0, 30.0)
        BMI = st.slider('vücut-kitle indeksi', 0.0, 67.1, 32.0)
        DPF = st.slider('DPF', 0.078, 2.42, 0.3725)
        age = st.slider('yaş', 21, 81, 29)

        user_data = {
            'pregnancies': pregnancies,
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'BMI': BMI,
            'DPF': DPF,
            'age': age
        }

        features = pd.DataFrame(user_data, index=[0])
        return features

    user_input = get_user_input()

    st.subheader('Kullanıcı Girdileri:')
    st.write(user_input)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    st.subheader('Model Test Doğruluk Yüzdesi:')
    st.write('%' + str(accuracy_score(y_test, clf.predict(X_test)) * 100))

    prediction = clf.predict(user_input)

    st.subheader('Tahmin Sonucu:')
    st.write(prediction)

    return None