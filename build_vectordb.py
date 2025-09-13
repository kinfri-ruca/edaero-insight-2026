import os
import json
import chromadb
import google.generativeai as genai

# --- 설정 ---
JSON_FILE_PATH = 'result_2026_서울대학교_정시.json'
DB_PATH = "chroma_db"
COLLECTION_NAME = "admissions_2026"
EMBEDDING_MODEL = 'models/text-embedding-004'
# ------------------------------------

# 1. API 키 설정
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini API 키가 성공적으로 설정되었습니다.")
except KeyError:
    print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요.")
    exit()

def build_vector_db():
    """JSON 파일을 읽어 ChromaDB에 벡터 데이터베이스를 구축합니다."""
    print(f"'{JSON_FILE_PATH}' 파일을 읽어옵니다...")
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 에러: '{JSON_FILE_PATH}' 파일을 찾을 수 없습니다.")
        return

    # ChromaDB 클라이언트 초기화 (파일 기반으로 데이터 저장)
    client = chromadb.PersistentClient(path=DB_PATH)

    # 컬렉션 생성 또는 기존 컬렉션 가져오기
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
    )

    print(f"Vector DB 컬렉션 '{COLLECTION_NAME}' 준비 완료. 데이터 임베딩 및 추가를 시작합니다...")

    documents = []
    metadatas = []
    ids = []

           # 2. 데이터를 문서, 메타데이터, ID로 변환
    for i, item in enumerate(data):
        # 검색의 대상이 될 텍스트 문서 (의미를 담고 있는 부분)
        content = (
            f"전형명: {item.get('selection_category') or ''}, "
            f"모집단위: {item.get('major') or ''}, "
            f"지원자격: {item.get('eligibility') or ''}, "
            f"전형방법: {str(item.get('evaluation_method') or '')}"
        )
        documents.append(content)

        # 검색 결과와 함께 제공될 추가 정보 (None 값을 ''로 변환)
        metadatas.append({
            "university": str(item.get('university') or ''),
            "year": str(item.get('year') or ''),
            "category": str(item.get('selection_category') or ''),
            "major": str(item.get('major') or ''),
            "source_page": str(item.get('source_page') or '')
        })

        # 각 항목의 고유 ID
        ids.append(f"item_{i}")

    # 3. 데이터 임베딩 및 DB에 추가
    # Gemini API를 사용하여 documents 리스트 전체를 한번에 임베딩
    embeddings = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=documents,
        task_type="retrieval_document" # 문서 검색용 임베딩
    )

    # --- 👇 여기에 디버깅 코드를 추가해주세요 👇 ---
    print("\n--- 디버깅 정보 ---")
    print(f"ID 리스트 길이: {len(ids)}")
    print(f"메타데이터 리스트 길이: {len(metadatas)}")
    print(f"문서 리스트 길이: {len(documents)}")
    print(f"임베딩 리스트 길이: {len(embeddings['embedding'])}")
    print("--------------------\n")

    collection.add(
        embeddings=embeddings['embedding'],
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"\n✨ Vector DB 구축 완료! 총 {collection.count()}개의 항목이 추가되었습니다.")

if __name__ == "__main__":
    build_vector_db()