from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import re
import os

# --- KESİN AYARLAR ---
# Terminal 'Proje' klasöründeyken bu yollar doğrudur
TARIFLER_DIR = "src/Tarifler" 
PERSIST_DIRECTORY = "src/chroma_db"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

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
    if not os.path.exists(TARIFLER_DIR):
        # Eğer src içindeysek yolu düzeltmeye çalış
        if os.path.exists("Tarifler"):
            base = "."
        else:
            print(f"❌ Hata: '{TARIFLER_DIR}' klasörü bulunamadı!")
            return
    else:
        base = "src"

    loader = DirectoryLoader(os.path.join(base, "Tarifler"), glob="*.txt")
    documents = loader.load()
    
    if not documents:
        print("⚠️ Tarif dosyası bulunamadı!")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=os.path.join(base, "chroma_db")
    )
    print("✅ Veritabanı başarıyla güncellendi.")

def yemek_tarifi_ajani(sorgu, max_sonuc=5):
    # Yol kontrolü
    db_yolu = PERSIST_DIRECTORY if os.path.exists("src") else "chroma_db"
    
    if not os.path.exists(db_yolu):
        veritabani_olustur()

    vectordb = Chroma(persist_directory=db_yolu, embedding_function=embeddings)
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
    veritabani_olustur()