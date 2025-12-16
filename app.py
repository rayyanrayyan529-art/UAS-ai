import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="UAS Kecerdasan Artifisial",
    layout="wide",
    page_icon="🤖"
)

# --- Sidebar Navigasi ---
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Beranda", "Upload Data", "Training Model", "Prediksi"]
)

# Variabel Global Sederhana (dalam session_state)
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = []
if "target" not in st.session_state:
    st.session_state.target = None

# --- HALAMAN BERANDA ---
if menu == "Beranda":
    st.title("🤖 UAS Kecerdasan Artifisial")
    st.subheader("Selamat datang di aplikasi Machine Learning berbasis Streamlit!")
    
    st.write("""
    Aplikasi ini dibuat untuk memenuhi tugas **UAS Kecerdasan Artifisial**.
    
    ### Fitur yang tersedia:
    - 📂 Upload Dataset CSV  
    - 📊 Preview Data  
    - ⚙️ Preprocessing  
    - 🧠 Training Model (KNN, Naive Bayes, Decision Tree, Random Forest)  
    - 📈 Evaluasi Model  
    - 🔮 Prediksi Input Data Baru  
    
    Silakan pilih menu di sidebar untuk memulai.
    """)

# --- HALAMAN UPLOAD DATA ---
elif menu == "Upload Data":
    st.title("📂 Upload Dataset")

    data = st.file_uploader("Upload file CSV", type=["csv"])

    if data:
        df = pd.read_csv(data)
        st.session_state.df = df

        st.success("Dataset berhasil diupload!")
        st.write("### Preview Data")
        st.dataframe(df)

        st.write("### Statistik Data")
        st.write(df.describe())

# --- HALAMAN TRAINING MODEL ---
elif menu == "Training Model":
    st.title("🧠 Training Model Machine Learning")

    if st.session_state.df is None:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu **Upload Data**.")
        st.stop()

    df = st.session_state.df

    # Pilih target dan fitur
    st.write("### Pilih target (label)")
    target = st.selectbox("Kolom Target", df.columns)

    st.write("### Pilih fitur")
    features = st.multiselect("Kolom Fitur", df.columns.drop(target))

    if features and target:
        X = df[features]
        y = df[target]

        # Splitting
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Pilih model
        st.write("### Pilih Model")
        model_name = st.selectbox(
            "Model",
            ["KNN", "Naive Bayes", "Decision Tree", "Random Forest"]
        )

        if st.button("🚀 Train Model"):
            if model_name == "KNN":
                model = KNeighborsClassifier()
            elif model_name == "Naive Bayes":
                model = GaussianNB()
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state.model = model
            st.session_state.features = features
            st.session_state.target = target

            st.success("Model berhasil ditraining!")
            st.write(f"### Akurasi Model: **{accuracy_score(y_test, y_pred):.2f}**")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # Classification report
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

# --- HALAMAN PREDIKSI ---
elif menu == "Prediksi":
    st.title("🔮 Prediksi Data Baru")

    if st.session_state.model is None:
        st.warning("⚠️ Silakan train model terlebih dahulu.")
        st.stop()

    model = st.session_state.model
    features = st.session_state.features

    st.write("### Masukkan nilai untuk prediksi:")

    input_data = []
    for f in features:
        val = st.number_input(f, value=0.0)
        input_data.append(val)

    if st.button("Prediksi"):
        arr = np.array([input_data])
        result = model.predict(arr)
        st.success(f"Hasil Prediksi: **{result[0]}**")
