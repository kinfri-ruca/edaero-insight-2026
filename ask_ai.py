import os
import json
import chromadb
import google.generativeai as genai

# --- 설정 ---
DB_PATH = "chroma_db"
STRUCTURED_COLLECTION = "structured_data"
RAW_COLLECTION = "raw_chunks_semantic"
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATIVE_MODEL = 'gemini-2.5-flash'
KEYWORD_MODEL = 'gemini-2.5-flash'
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

def main():
    """하이브리드 검색(Python 필터링+벡터)을 사용하는 챗봇 메인 함수"""
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        structured_collection = client.get_collection(name=STRUCTURED_COLLECTION)
        raw_collection = client.get_collection(name=RAW_COLLECTION)
        # (NEW) 시작할 때 구조화된 DB 전체를 미리 불러옵니다.
        print("사전 데이터 로딩 중...")
        all_structured_data = structured_collection.get()
        print(f"✅ 총 {len(all_structured_data['ids'])}개의 구조화된 데이터 로딩 완료.")
    except Exception as e:
        print(f"❌ DB 컬렉션 연결 실패. DB가 올바르게 구축되었는지 확인해주세요.")
        print(f"   오류: {e}")
        return
    
    print("\n--- edaeroAI 챗봇 (Final Ver.) ---")
    print("안녕하세요! edaeroAI - 대학추천 AI 컨설턴트입니다.")
    print("입시 요강에 대해 무엇이든 물어보세요.")
    print("종료하려면 'exit' 또는 '종료'를 입력하세요.")

    while True:
        query = input("\n[질문]: ")
        if query.lower() in ['exit', '종료']:
            print("챗봇을 종료합니다. 이용해주셔서 감사합니다.")
            break

        # 1. Gemini를 이용해 질문에서 '학과명' 키워드 추출
        keyword_extractor_model = genai.GenerativeModel(KEYWORD_MODEL)
        keyword_prompt = f"다음 질문에서 대학 '학과명' 또는 '전공명'을 정확히 하나만 추출해줘. 만약 학과명이 언급되지 않았다면, '없음'이라고만 대답해줘. 질문: \"{query}\""
        response = keyword_extractor_model.generate_content(keyword_prompt)
        major_keyword = response.text.strip().replace(".", "")
        print(f"--- [디버깅] 추출된 학과 키워드: {major_keyword} ---")

        # 2. 키워드 존재 여부에 따라 검색 전략 변경
        query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=query, task_type="retrieval_query")['embedding']

        if major_keyword != "없음":
            print("--- [디버깅] '정확 검색(Python 필터링)'을 수행합니다. ---")
            
            # (FIXED) 미리 불러온 전체 데이터에서 Python으로 직접 필터링
            matching_docs = []
            matching_metas = []
            matching_ids = []
            for i, meta in enumerate(all_structured_data['metadatas']):
                 if major_keyword.replace(" ", "") in (meta.get('major') or '').replace(" ", ""):
                    matching_docs.append(all_structured_data['documents'][i])
                    matching_metas.append(meta)
                    matching_ids.append(all_structured_data['ids'][i])
            
            # ChromaDB의 query 결과와 동일한 형식으로 맞춰줍니다.
            structured_results = {
                'ids': [matching_ids],
                'documents': [matching_docs],
                'metadatas': [matching_metas]
            }
            raw_results = raw_collection.query(query_embeddings=[query_embedding], n_results=3)
        else:
            print("--- [디버깅] '유사도 검색(벡터 검색)'을 수행합니다. ---")
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
        
        # 4. 최종 답변 생성을 위한 프롬프트 구성 (이전과 동일)
        prompt = f"""
        당신은 대한민국 최고의 입시 전문가 'edaeroAI'이며, 탐정처럼 주어진 정보를 분석하여 질문에 답해야 한다.
        **[임무]** 주어진 '[참고 자료]'를 바탕으로 사용자의 '[질문]'에 대해 가장 정확한 답변을 찾아내라.
        **[작업 절차]**
        1. **타겟 확정:** 사용자의 '[질문]'에서 핵심적인 대상(학과, 전형 이름 등)이 무엇인지 명확히 파악한다. 현재 질문의 핵심 타겟은 '{query}' 이다.
        2. **자료 수색:** '[참고 자료]' 전체를 꼼꼼히 읽는다. 특히 '[핵심 요약 정보]' 섹션을 주목한다.
        3. **증거 확보:** 참고 자료의 각 항목 중에서, 1단계에서 파악한 **핵심 타겟과 정확히 일치하거나 가장 직접적으로 관련된 정보**를 찾는다. 관련 없는 정보는 과감히 무시한다.
        4. **최종 보고:** 3단계에서 확보한 증거만을 바탕으로 질문에 대한 답변을 생성한다. 만약 여러 증거를 확인했음에도 타겟과 관련된 정보를 정말 찾을 수 없다면, "제공된 자료에서는 해당 정보를 찾을 수 없었습니다."라고 보고한다.
        5. 모든 답변의 근거가 된 출처를 명확히 밝힌다.

        [참고 자료]
        {context}
        [질문]
        {query}
        [탐정 edaeroAI의 최종 보고서]
        """
        
        # 5. Gemini 모델을 통해 최종 답변 생성
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        final_response = model.generate_content(prompt)

        print("\n[edaeroAI]:")
        print(final_response.text)
        if sources:
            print("\n" + "\n".join(list(set(sources))))

if __name__ == "__main__":
    if initialize_services():
        main()