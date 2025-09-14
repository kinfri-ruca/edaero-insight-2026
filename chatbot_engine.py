import os
import json
import chromadb
import google.generativeai as genai
import streamlit as st

# --- 설정 ---
DB_PATH = "chroma_db"
STRUCTURED_COLLECTION = "structured_data"
RAW_COLLECTION = "raw_chunks_semantic"
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATIVE_MODEL = 'gemini-2.5-flash'
KEYWORD_MODEL = 'gemini-2.5-flash'
# ------------------------------------

# Streamlit의 캐싱 기능으로 AI 모델과 DB를 한번만 로드하게 하여 속도를 높입니다.
#@st.cache_resource
def load_ai_resources():
    """앱 실행 시 한번만 AI 리소스를 로드하는 함수"""
    print("AI 리소스를 로딩합니다...")
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
    except KeyError:
        st.error("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다! 터미널을 재시작하고 다시 설정해주세요.")
        return None, None, None

    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        structured_collection = client.get_collection(name=STRUCTURED_COLLECTION)
        raw_collection = client.get_collection(name=RAW_COLLECTION)
        print("✅ DB 컬렉션 연결 완료.")
    except Exception as e:
        st.error(f"DB 컬렉션 연결에 실패했습니다: {e}")
        return None, None, None
        
    print("✅ AI 리소스 로딩 완료.")
    return structured_collection, raw_collection, genai

def get_ai_response(query, structured_collection, raw_collection, genai_alias):
    """사용자의 질문에 대한 AI의 최종 답변을 생성합니다."""
    # 1. Gemini를 이용해 질문에서 '학과명' 키워드 추출
    keyword_extractor_model = genai_alias.GenerativeModel(KEYWORD_MODEL)
    keyword_prompt = f"다음 질문에서 대학 '학과명' 또는 '전공명'을 정확히 하나만 추출해줘. 만약 학과명이 언급되지 않았다면, '없음'이라고만 대답해줘. 질문: \"{query}\""
    response = keyword_extractor_model.generate_content(keyword_prompt)
    major_keyword = response.text.strip().replace(".", "")

    # 2. 키워드 존재 여부에 따라 검색 전략 변경
    if major_keyword != "없음":
        # '정확 검색': 메타데이터 필터링
        all_data = structured_collection.get() # .get()은 전체 데이터를 가져옴
        matching_docs = []
        matching_metas = []
        matching_ids = []
        for i, meta in enumerate(all_data['metadatas']):
            if major_keyword.replace(" ", "") in (meta.get('major') or '').replace(" ", ""):
                matching_docs.append(all_data['documents'][i])
                matching_metas.append(meta)
                matching_ids.append(all_data['ids'][i])
        structured_results = {'ids': [matching_ids], 'documents': [matching_docs], 'metadatas': [matching_metas]}
        
        query_embedding = genai_alias.embed_content(model=EMBEDDING_MODEL, content=query, task_type="retrieval_query")['embedding']
        raw_results = raw_collection.query(query_embeddings=[query_embedding], n_results=3)
    else:
        # '유사도 검색': 벡터 검색
        query_embedding = genai_alias.embed_content(model=EMBEDDING_MODEL, content=query, task_type="retrieval_query")['embedding']
        structured_results = structured_collection.query(query_embeddings=[query_embedding], n_results=10)
        raw_results = raw_collection.query(query_embeddings=[query_embedding], n_results=5)

    # 3. 검색된 모든 정보를 종합하여 '참고 자료' 생성
    context = "--- [핵심 요약 정보 (구조화된 데이터)] ---\n"
    sources = []
    if structured_results['metadatas'] and structured_results['metadatas'][0]:
        for meta in structured_results['metadatas'][0]:
            context += json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
            sources.append(f"출처 (정형 데이터): {meta.get('major', '정보')} (p.{meta.get('source_page', 'N/A')})")

    context += "\n\n--- [관련 원본 텍스트 (추가 정보)] ---\n"
    if raw_results['documents'] and raw_results['documents'][0]:
        context += "\n".join(raw_results['documents'][0])
        for meta in raw_results['metadatas'][0]:
             sources.append(f"출처 (원본 텍스트): {meta.get('source_info', 'N/A')}")
    
    # 4. 최종 답변 생성을 위한 '탐정 프롬프트' 구성
    prompt = f"""
    당신은 대한민국 최고의 입시 전문가 'edaeroAI'이며, 탐정처럼 주어진 정보를 분석하여 질문에 답해야 한다.

    **[임무]**
    주어진 '[참고 자료]'를 바탕으로 사용자의 '[질문]'에 대해 가장 정확한 답변을 찾아내라.

    **[작업 절차]**
    1.  **타겟 확정:** 사용자의 '[질문]'에서 핵심적인 대상(학과, 전형 이름 등)이 무엇인지 명확히 파악한다. 현재 질문의 핵심 타겟은 '{query}' 이다.
    2.  **자료 수색:** '[참고 자료]' 전체를 꼼꼼히 읽는다. 특히 '[핵심 요약 정보]' 섹션을 주목한다.
    3.  **증거 확보:** 참고 자료의 각 항목 중에서, 1단계에서 파악한 **핵심 타겟과 정확히 일치하거나 가장 직접적으로 관련된 정보**를 찾는다. 관련 없는 정보는 과감히 무시한다.
    4.  **최종 보고:** 3단계에서 확보한 증거만을 바탕으로 질문에 대한 답변을 생성한다. 만약 여러 증거를 확인했음에도 타겟과 관련된 정보를 정말 찾을 수 없다면, "제공된 자료에서는 해당 정보를 찾을 수 없었습니다."라고 보고한다.
    5.  모든 답변의 근거가 된 출처를 명확히 밝힌다.

    [참고 자료]
    {context}
    [질문]
    {query}
    [탐정 edaeroAI의 최종 보고서]
    """
    
    # 5. Gemini 모델을 통해 최종 답변 생성
    model = genai_alias.GenerativeModel(GENERATIVE_MODEL)
    final_response = model.generate_content(prompt)
    
    return final_response.text, list(set(sources))