
import streamlit as st
from dataset_page import show_dataset_page
from cleandata_page import show_cleandata_page
from models_page import show_models_page
from predict_page import show_predict_page


page = st.sidebar.selectbox("Gitmek istediğin sayfayı seç", ("Giriş ve Veri Setinin İlk Hali", "Makine Öğrenmesi Algoritmaları",
                                                             "Düzenlenmiş Veri Seti ile Sonuçlar", "Diyabet Tahmini"))

if page == "Giriş ve Veri Setinin İlk Hali":
    show_dataset_page()
elif page == "Makine Öğrenmesi Algoritmaları":
    show_models_page()
elif page == "Düzenlenmiş Veri Seti ile Sonuçlar":
    show_cleandata_page()
else:
    show_predict_page()