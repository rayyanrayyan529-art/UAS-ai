import streamlit as st

st.title("UAS AI - Aplikasi Klasifikasi Sederhana")
st.write("Simulasi klasifikasi tanpa library berat (aman Streamlit Cloud)")

st.subheader("Input Data")

umur = st.number_input("Umur", min_value=18, max_value=70, value=30)
gaji = st.number_input("Gaji (juta)", min_value=1, max_value=20, value=5)

if st.button("Prediksi"):
    # RULE SEDERHANA (AI BERBASIS ATURAN)
    if umur >= 30 and gaji >= 6:
        st.success("✅ DITERIMA")
    else:
        st.error("❌ TIDAK DITERIMA")

st.caption("Model: Rule-Based Artificial Intelligence")

# =========================
# LOAD DATASET
# =========================
iris = load_iris()
X = iris.data
y = iris.target

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

st.success(f"Akurasi Model: {akurasi:.2f}")

# =========================
# INPUT USER
# =========================
st.subheader("Prediksi Data Baru")

sepal_length = st.number_input("Sepal Length", value=5.1)
sepal_width  = st.number_input("Sepal Width", value=3.5)
petal_length = st.number_input("Petal Length", value=1.4)
petal_width  = st.number_input("Petal Width", value=0.2)

if st.button("Prediksi"):
    hasil = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write("Hasil Prediksi:", iris.target_names[hasil[0]])

