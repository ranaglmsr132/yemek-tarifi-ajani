import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import re
import os

# --- AYARLAR ---
# GitHub yapınıza göre 'kaynak' klasörü kullanılıyor
TARIFLER_DIR = "kaynak" 
PERSIST_DIRECTORY = "chroma_db"

# Embeddings modelini bir kez tanımlıyoruz
@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

STOP_KELIMELER = {
    "olan", "tarif", "tarifleri", "tarifler", "yemek", "yemekler",
    "öner", "önerir", "önerisi", "bul", "bulur", "getir",
    "ne", "ile", "içinde", "kullanılan", "kullan",
    "bana", "bir", "mi", "mu", "mı", "var", "listele"
}

def anahtar_kelimeleri_cikar(sorgu):
    kelimeler = re.findall(r"[a-zçğıöşü]+", sorgu.lower())
    return [k for k in kelimeler if k not in STOP_KELIMELER and len(k) > 2]

def veritabani_olustur():
    documents = []
    
    # 1. Klasör ismini buraya yazın (GitHub'da gördüğünüzün aynısı olmalı)
    hedef_klasor = "src" 

    # Mevcut dizindeki tüm dosyaları ve klasörleri listele (Hata ayıklama için)
    mevcut_dosyalar = os.listdir(".")
    st.write(f"Ana dizindeki dosyalar: {mevcut_dosyalar}") # Bu satır klasörün adını görmemizi sağlar

    # Yol tespiti
    if os.path.exists(hedef_klasor):
        path_to_check = hedef_klasor
    elif os.path.exists(os.path.join("src", hedef_klasor)):
        path_to_check = os.path.join("src", hedef_klasor)
    else:
        st.error(f"❌ '{hedef_klasor}' klasörü hiçbir yerde bulunamadı!")
        st.info(f"Sistemdeki mevcut dosyalar: {mevcut_dosyalar}")
        return

    try:
        # 2. DOSYALARI YÜKLE
        loader = DirectoryLoader(path_to_check, glob="**/*.txt")
        documents = loader.load()
    except Exception as e:
        st.error(f"Yükleme hatası: {e}")
        return
    
    if not documents:
        st.warning(f"⚠️ '{path_to_check}' klasörü bulundu ama içinde .txt dosyası yok!")
        return

    # 3. VERİTABANI OLUŞTURMA
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(documents)
    Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
    st.success("✅ Veritabanı başarıyla güncellendi.")
def yemek_tarifi_ajani(sorgu, max_sonuc=5):
    # Eğer veritabanı klasörü yoksa oluştur
    if not os.path.exists(PERSIST_DIRECTORY):
        veritabani_olustur()

    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    docs = vectordb.similarity_search(sorgu, k=20)
    
    arananlar = anahtar_kelimeleri_cikar(sorgu)
    kesin_sonuclar = []
    
    for doc in docs:
        metin = doc.page_content.lower()
        if all(kelime in metin for kelime in arananlar):
            if doc.page_content not in [d.page_content for d in kesin_sonuclar]:
                kesin_sonuclar.append(doc)
    
    return kesin_sonuclar[:max_sonuc]

if __name__ == "__main__":
    # Direkt script olarak çalıştırıldığında (opsiyonel)
    veritabani_olustur()
# --- KULLANICI ARAYÜZÜ (STREAMLIT) ---
st.title("👨‍🍳 Yapay Zeka Yemek Tarifi Asistanı")
st.markdown("Elinizdeki malzemeleri yazın veya bir yemek adı arayın!")

# --- KULLANICI ARAYÜZÜ (STREAMLIT) ---
# Üstteki teknik bilgileri (Ana dizindeki dosyalar vb.) görmek istemediğin için 
# veritabani_olustur() içindeki st.write ve st.success satırlarını silebilir 
# veya aşağıdaki gibi arayüzü temiz tutabilirsin.

st.title("👨‍🍳 Yemek Tarifi Asistanı")
st.markdown("Merhabalar! Bugün size hangi yemeği hazırlamamda yardımcı olabilirim?")

# Kullanıcıdan girdi al
sorgu = st.text_input("Mesajınızı yazın:", placeholder="Örn: İçinde domates olan tarifleri listeler misin?")

if sorgu:
    with st.spinner("Sizin için tariflerimi kontrol ediyorum..."):
        sonuclar = yemek_tarifi_ajani(sorgu)
        
        if sonuclar:
            # Chatbot yanıtı gibi bir giriş metni
            st.markdown(f"### 🤖 Asistanın Yanıtı:")
            st.write(f"Harika bir seçim! Aradığınız kriterlere uygun **{len(sonuclar)} adet** tarif buldum. İşte detaylar:")
            
            # Sonuçları düz metin (text) olarak göster
            for i, doc in enumerate(sonuclar):
                st.markdown(f"---")
                st.markdown(f"#### 📝 Tarif {i+1}")
                st.text(doc.page_content) # expaner yerine direkt text formatında gösterir
        else:
            st.markdown("### 🤖 Asistanın Yanıtı:")
            st.write("Üzgünüm, tarif defterimde buna uygun bir kayıt bulamadım. Başka bir malzeme veya yemek ismi denemek ister misiniz?")
