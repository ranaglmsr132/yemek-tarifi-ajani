import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import re
import os

# --- AYARLAR ---
# GitHub yapındaki klasör yolu (src/kaynak)
TARIFLER_DIR = "src/kaynak" 
PERSIST_DIRECTORY = "chroma_db"

# Embeddings modelini önbelleğe alarak yüklüyoruz
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
    """Arka planda veritabanını oluşturur, kullanıcıya teknik detay göstermez."""
    documents = []
    
    # Klasör kontrolü
    if not os.path.exists(TARIFLER_DIR):
        # Eğer direkt 'kaynak' olarak ana dizindeyse onu dene
        if os.path.exists("kaynak"):
            path_to_check = "kaynak"
        else:
            return # Sessizce hata yönetimini chatbot içinde yapacağız
    else:
        path_to_check = TARIFLER_DIR

    try:
        loader = DirectoryLoader(path_to_check, glob="**/*.txt")
        documents = loader.load()
        
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = splitter.split_documents(documents)
            Chroma.from_documents(
                documents=texts, 
                embedding=embeddings, 
                persist_directory=PERSIST_DIRECTORY
            )
    except:
        pass # Teknik hataları kullanıcı arayüzüne basmıyoruz

def yemek_tarifi_ajani(sorgu, max_sonuc=5):
    # Veritabanı yoksa oluştur
    if not os.path.exists(PERSIST_DIRECTORY):
        veritabani_olustur()

    try:
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
    except:
        return []

# --- KULLANICI ARAYÜZÜ (STREAMLIT CHATBOT) ---
st.set_page_config(page_title="Yemek Asistanı", page_icon="👨‍🍳")

st.title("👨‍🍳 Yemek Tarifi Asistanı")
st.markdown("Merhaba! Ben senin dijital şefinim. Elindeki malzemeleri söylersen sana en uygun tarifleri bulabilirim.")

# Kullanıcıdan mesaj al
sorgu = st.text_input("Mesajınızı yazın:", placeholder="Örn: İçinde domates olan tarifler...")

if sorgu:
    with st.spinner("Tarif defterimi karıştırıyorum..."):
        sonuclar = yemek_tarifi_ajani(sorgu)
        
        st.markdown("### 🤖 Şefin Yanıtı:")
        
        if sonuclar:
            st.write(f"Harika bir seçim! Aradığın kriterlere uygun **{len(sonuclar)} tarif** buldum:")
            
            for i, doc in enumerate(sonuclar):
                st.markdown(f"---")
                st.markdown(f"**📖 Seçenek {i+1}**")
                # Tarif içeriğini temiz metin olarak gösteriyoruz
                st.info(doc.page_content)
        else:
            st.write("Üzgünüm, tarif defterimde buna uygun tam bir eşleşme bulamadım. Malzemeleri değiştirmeyi veya daha genel aramayı deneyebilir misin?")

if __name__ == "__main__":
    # Veritabanı yoksa ilk seferde sessizce oluşturur
    if not os.path.exists(PERSIST_DIRECTORY):
        veritabani_olustur()
