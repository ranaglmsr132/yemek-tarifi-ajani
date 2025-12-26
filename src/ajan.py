import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import re
import os

# --- AYARLAR ---
HEDEF_KLASOR = "Tarifler"
PERSIST_DIRECTORY = "chroma_db"

@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()
#Gürültüyü Temizleme
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
    
    if os.path.exists(HEDEF_KLASOR):
        path_to_check = HEDEF_KLASOR
    elif os.path.exists(os.path.join("src", HEDEF_KLASOR)):
        path_to_check = os.path.join("src", HEDEF_KLASOR)
    else:
      
        st.error(f"❌ '{HEDEF_KLASOR}' klasörü GitHub'da bulunamadı!")
        return

    try:
        
        loader = DirectoryLoader(path_to_check, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = splitter.split_documents(documents)
            Chroma.from_documents(
                documents=texts, 
                embedding=embeddings, 
                persist_directory=PERSIST_DIRECTORY
            )
            st.success(f"✅ {len(documents)} adet tarif başarıyla yüklendi!")
    except Exception as e:
        st.error(f"Yükleme hatası: {e}")

def yemek_tarifi_ajani(sorgu, max_sonuc=5):
    if not os.path.exists(PERSIST_DIRECTORY):
        veritabani_olustur()

    try:
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        docs = vectordb.similarity_search(sorgu, k=5)
        
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

# --- CHATBOT ARAYÜZÜ ---
st.set_page_config(page_title="Şef Asistan", page_icon="👩‍🍳")

st.title("👩‍🍳 Akıllı Yemek Asistanı")
st.write("Tarif defterimdeki verilerle size yardımcı olmaya hazırım.")

sorgu = st.text_input("Bugün ne pişirelim?", placeholder="Örn: Tavuklu bir tarif var mı?")

if sorgu:
    with st.spinner("Tariflerimi inceliyorum..."):
        sonuclar = yemek_tarifi_ajani(sorgu)
        
        st.markdown("### 🤖 Şefin Önerisi:")
        if sonuclar:
            for i, doc in enumerate(sonuclar):
                st.markdown(f"**Seçenek {i+1}**")
                st.info(doc.page_content)
                st.markdown("---")
        else:
            st.warning("Aradığınız kriterlere uygun bir tarif bulamadım. Lütfen farklı malzemeler yazmayı deneyin.")

if __name__ == "__main__":
    
    if not os.path.exists(PERSIST_DIRECTORY):
        veritabani_olustur()
