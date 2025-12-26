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
    
    # Klasör kontrolü
    if not os.path.exists(TARIFLER_DIR):
        st.error(f"❌ Hata: '{TARIFLER_DIR}' klasörü bulunamadı! Lütfen GitHub'da bu klasörün olduğundan emin olun.")
        return

    try:
        # Klasördeki .txt dosyalarını yükle
        loader = DirectoryLoader(TARIFLER_DIR, glob="**/*.txt")
        documents = loader.load()
    except Exception as e:
        st.error(f"Dosyalar yüklenirken hata oluştu: {e}")
        return
    
    if not documents:
        st.warning("⚠️ Klasör boş veya içinde .txt dosyası bulunamadı!")
        return

    # Metin parçalama
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(documents)

    # Chroma veritabanını oluştur ve kaydet
    Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
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
