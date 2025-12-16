import streamlit as st

# =========================
# JUDUL APLIKASI
# =========================
st.title("UAS AI - Sistem Klasifikasi Rule-Based")
st.write("Aplikasi Artificial Intelligence sederhana berbasis aturan")

# =========================
# INPUT DATA
# =========================
st.subheader("Input Data")

umur = st.number_input("Masukkan Umur", min_value=18, max_value=70, value=30)
gaji = st.number_input("Masukkan Gaji (juta)", min_value=1, max_value=20, value=5)

# =========================
# RULE-BASED AI
# =========================
if st.button("Prediksi"):
    if umur >= 30 and gaji >= 6:
        st.success("✅ HASIL: DITERIMA")
    else:
        st.error("❌ HASIL: TIDAK DITERIMA")

# =========================
# KETERANGAN
# =========================
st.caption("Metode: Rule-Based Artificial Intelligence")
