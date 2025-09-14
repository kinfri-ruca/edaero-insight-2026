# app.py
import streamlit as st
import time
import chatbot_engine # 우리가 만든 AI 엔진을 불러옵니다.

st.set_page_config(page_title="edaeroAI", layout="wide")

st.title("🤖 edaeroAI - 대학추천 AI 컨설턴트")

# AI 리소스 로딩 (앱 실행 시 한번만)
structured_collection, raw_collection, genai_alias = chatbot_engine.load_ai_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("입시 요강에 대해 무엇이든 물어보세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI에게 실제 답변 요청
    with st.chat_message("assistant"):
        with st.spinner("edaeroAI가 분석 중입니다..."):
            # AI 엔진 호출
            response, sources = chatbot_engine.get_ai_response(prompt, structured_collection, raw_collection, genai_alias)
            
            # 출처 정보가 있다면 답변에 추가
            if sources:
                response += "\n\n--- \n**참고 자료:**\n"
                for source in sources:
                    response += f"- {source}\n"
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})