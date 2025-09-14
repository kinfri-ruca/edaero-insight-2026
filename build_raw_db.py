# build_raw_db.py
import os
import re
import chromadb
import google.generativeai as genai

# --- 설정 ---
RAW_TEXT_FILE = 'result_2026_서울시립대학교_정시_raw_text_v4.txt' # main.py 실행 후 생성된 파일
DB_PATH = "chroma_db"
COLLECTION_NAME = "raw_chunks" # 새로운 컬렉션 이름
EMBEDDING_MODEL = 'models/text-embedding-004'
# ------------------------------------

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요.")
    exit()

def build_raw_chunks_db():
    print(f"'{RAW_TEXT_FILE}' 파일을 읽어 원본 텍스트 DB를 구축합니다...")
    with open(RAW_TEXT_FILE, 'r', encoding='utf-8') as f:
        full_text = f.read()
        
    pages = re.split(r'--- Page \d+ ---', full_text)
    documents = [p.strip() for p in pages if p.strip()]
    metadatas = [{"source_page": i+1} for i in range(len(documents))]
    ids = [f"page_{i+1}" for i in range(len(documents))]

    client = chromadb.PersistentClient(path=DB_PATH)
    # get_or_create_collection을 사용하면 기존에 있어도 오류 없이 가져옵니다.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    print(f"'{COLLECTION_NAME}' 컬렉션에 페이지별 텍스트 임베딩 및 추가를 시작합니다...")
    embeddings = genai.embed_content(
        model=EMBEDDING_MODEL, content=documents, task_type="retrieval_document"
    )
    
    # id가 이미 존재할 경우를 대비해 upsert 사용
    collection.upsert(
        embeddings=embeddings['embedding'], documents=documents, metadatas=metadatas, ids=ids
    )
    
    print(f"✨ '{COLLECTION_NAME}' DB 구축 완료! 총 {collection.count()}개의 페이지가 저장되었습니다.")

if __name__ == "__main__":
    build_raw_chunks_db()