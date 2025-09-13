import os
import chromadb
import google.generativeai as genai

# --- 설정 ---
DB_PATH = "chroma_db"
COLLECTION_NAME = "admissions_2026"
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATIVE_MODEL = 'gemini-1.5-pro-latest'
# ------------------------------------

# 1. API 키 설정
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini API 키가 성공적으로 설정되었습니다.")
except KeyError:
    print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요.")
    exit()

def main():
    """사용자 질문에 답변하는 챗봇 메인 함수"""
    # Vector DB 클라이언트 초기화 및 컬렉션 가져오기
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    print("\n--- edaeroAI 챗봇 ---")
    print("안녕하세요! edaeroAI - 대학추천 AI 컨설턴트입니다.")
    print("2026학년도 서울대학교 정시모집에 대해 무엇이든 물어보세요.")
    print("종료하려면 'exit' 또는 '종료'를 입력하세요.")

    while True:
        # 1. 사용자 질문 입력
        query = input("\n[질문]: ")
        if query.lower() in ['exit', '종료']:
            print("챗봇을 종료합니다. 이용해주셔서 감사합니다.")
            break

        # 2. 질문을 벡터로 변환 (임베딩)
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query" # '질문'용 임베딩
        )['embedding']

        # 3. DB에서 가장 유사한 정보 검색 (3개)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # 4. 검색된 정보를 바탕으로 AI에게 전달할 '참고 자료' 생성
        context = ""
        sources = []
        for i, doc in enumerate(results['documents'][0]):
            context += f"\n--- 참고자료 {i+1} ---\n"
            context += doc
            sources.append(f"출처: {results['metadatas'][0][i]['university']} {results['metadatas'][0][i]['category']} (p.{results['metadatas'][0][i]['source_page']})")

        # 5. 최종 답변 생성을 위한 프롬프트 구성
        prompt = f"""
        당신은 'edaeroAI - 대학추천 AI 컨설턴트'입니다.
        아래에 제공된 '참고 자료'를 바탕으로 사용자의 '질문'에 대해 가장 정확하고 친절하게 답변해주세요.
        답변은 반드시 '참고 자료'에 있는 내용만을 근거로 생성해야 합니다. 자료에 없는 내용은 '알 수 없음'이라고 답변하세요.
        답변 마지막에는 어떤 자료를 참고했는지 출처를 명확히 밝혀주세요.

        [참고 자료]
        {context}

        [질문]
        {query}

        [답변]
        """

        # 6. Gemini 모델을 통해 최종 답변 생성
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        final_response = model.generate_content(prompt)

        print("\n[edaeroAI]:")
        print(final_response.text)
        print("\n" + "\n".join(list(set(sources)))) # 중복 제거 후 출처 표시

if __name__ == "__main__":
    main()