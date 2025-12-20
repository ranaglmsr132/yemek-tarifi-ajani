import streamlit as st
import os
import sys

# src klasörünü Python'un bulabilmesi için sisteme ekliyoruz
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from ajan import yemek_tarifi_ajani
except ImportError:
    from src.ajan import yemek_tarifi_ajani

st.set_page_config(page_title="Yemek Tarifi Ajanı", page_icon="👨‍🍳")
st.title("👨‍🍳 Akıllı Yemek Tarifi Ajanı")

sorgu = st.text_input("Sorgunuzu yazın (Örn: domates)")

if st.button("Tarifleri Getir"):
    if sorgu.strip():
        sonuclar = yemek_tarifi_ajani(sorgu)
        if sonuclar:
            st.success(f"{len(sonuclar)} tarif bulundu.")
            for i, doc in enumerate(sonuclar):
                with st.expander(f"📖 Tarif {i+1}", expanded=True):
                    st.write(doc.page_content)
        else:
            st.error("❌ Uygun tarif bulunamadı. Lütfen kelimeyi kontrol edin.")
    else:
        st.warning("Lütfen bir kelime girin.")