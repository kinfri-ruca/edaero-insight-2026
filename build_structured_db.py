import os
import json
import chromadb
import google.generativeai as genai

# --- 설정 ---
# main.py를 통해 최종 생성된 JSON 파일의 정확한 이름을 입력해주세요.
FINAL_JSON_FILE = 'result_2026_서울시립대학교_정시_final.json' 
DB_PATH = "chroma_db"
COLLECTION_NAME = "structured_data" # 구조화된 데이터 전용 컬렉션
EMBEDDING_MODEL = 'models/text-embedding-004'
# ------------------------------------

def initialize_services():
    """API 키 서비스를 초기화합니다."""
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
        print("✅ Gemini API 키가 성공적으로 설정되었습니다.")
        return True
    except KeyError:
        print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요.")
        return False
    
# build_structured_db.py 파일에서 이 함수 전체를 교체해주세요.

def build_structured_db():
    """JSON 파일을 읽어 '시맨틱 컨텍스트'를 포함한 구조화된 DB를 구축합니다."""
    print(f"'{FINAL_JSON_FILE}' 파일을 읽어 구조화된 DB를 구축합니다...")
    try:
        with open(FINAL_JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 에러: '{FINAL_JSON_FILE}' 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.")
        return

    department_info = data.get("department_info", [])
    if not department_info:
        print("❌ JSON 파일에서 'department_info' 데이터를 찾을 수 없습니다.")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"기존 '{COLLECTION_NAME}' 컬렉션을 삭제했습니다.")
    collection = client.create_collection(name=COLLECTION_NAME)
    
    documents, metadatas, ids = [], [], []
    for i, item in enumerate(department_info):
    # 검색을 위한 '핵심 키워드'만으로 document를 구성합니다.
        content = (
            f"학과명: {item.get('major') or ''}. "
            f"모집단위: {item.get('recruitment_unit') or ''}. "
            f"전형 종류: {item.get('selection_category') or ''}."
        )
        documents.append(content)
        
        # 답변 생성을 위한 전체 데이터는 metadata에 보관합니다.
        safe_item = {str(k): str(v or '') for k, v in item.items()}
        metadatas.append(safe_item)
        ids.append(f"dept_{i}")

    print(f"'{COLLECTION_NAME}' 컬렉션에 총 {len(documents)}개의 데이터 임베딩 및 추가를 시작합니다...")
    
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_documents = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        embeddings = genai.embed_content(
            model=EMBEDDING_MODEL, content=batch_documents, task_type="retrieval_document"
        )
        
        collection.add(
            embeddings=embeddings['embedding'], 
            documents=batch_documents, 
            metadatas=batch_metadatas, 
            ids=batch_ids
        )
        print(f"  - {i+len(batch_documents)}/{len(documents)}개 문서 처리 완료...")

    print(f"\n✨ '{COLLECTION_NAME}' DB 구축 완료! 총 {collection.count()}개의 학과 정보가 저장되었습니다.")

if __name__ == "__main__":
    if initialize_services():
        build_structured_db()