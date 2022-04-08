
# giriş sayfası ve veri seti görselleştirme

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_dataset_page():
    st.title('Makine Öğrenmesi Grafik Arayüzü')

    st.write("""
    ## DİYABET TAHMİNİ
    Makine Öğrenmesi ve Python kullanarak;
    *veri setinin ve kullanıcı girdilerin görselleştirilmesi,*
    *seçilen makine öğrenmesi algoritmasına göre sonuçların ve performans metriklerinin görselleştirilmesi,*
    *kullanıcı girdilerine göre diyabet olup olmadığını tahmin etmek.*
    """)

    df = pd.read_csv('diabetes.csv')

    st.subheader('Veri Setinin Görselleştirilmesi:')
    st.dataframe(df)
    st.write(df.describe())
    st.bar_chart(df)

    data = df["Pregnancies"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # equal daire olarak çizdirmek için

    st.write("""#### Veri Setindeki Hamilelik Sayısı Dağılımı""")

    st.pyplot(fig1)

    st.write(
        """
    #### Hamilelik Sayısı-Diyabet İlişkisi
    """
    )

    #data = df.groupby(["Pregnancies"]).count()

    data = df.groupby(["Pregnancies"])["Outcome"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    data = df["Age"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # daire olarak çizdirmek için

    st.write("""#### Veri Setindeki Yaş Dağılımı""")

    st.pyplot(fig1)

    st.write(
    """
    #### Farklı Yaşlarda Ortalama Diyabet Çıktısı
    """
    )

    data = df.groupby(["Age"])["Outcome"].mean().sort_values(ascending=True)
    st.line_chart(data)

    return None