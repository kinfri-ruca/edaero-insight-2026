# app.py

import streamlit as st
import chatbot_engine

# --- 설정 ---
PDF_URL = "https://firebasestorage.googleapis.com/v0/b/edaero-insight-2026.firebasestorage.app/o/2026_%EC%84%9C%EC%9A%B8%EC%8B%9C%EB%A6%BD%EB%8C%80%ED%95%99%EA%B5%90_%EC%A0%95%EC%8B%9C.pdf?alt=media&token=ccf53490-8cdd-469e-ae70-47ead5664dbc"
# -----------------------------

st.set_page_config(page_title="edaeroAI", layout="wide")
st.title("🤖 edaeroAI - 대학추천 AI 컨설턴트")

# AI 리소스 로딩
structured_collection, raw_collection, genai_alias = chatbot_engine.load_ai_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("입시 요강에 대해 무엇이든 물어보세요."):
    if not all([structured_collection, raw_collection, genai_alias]):
        st.error("AI 리소스 로딩에 실패했습니다. 환경 변수와 DB 경로를 확인해주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("edaeroAI가 분석 중입니다..."):
                response_stream, sources = chatbot_engine.get_ai_response(prompt, structured_collection, raw_collection, genai_alias)
                
                # --- 👇 여기가 핵심적인 변경 사항입니다! 👇 ---
                # st.write_stream 대신 수동으로 스트림을 처리합니다.
                message_placeholder = st.empty()
                full_response = ""
                try:
                    for chunk in response_stream:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"답변을 스트리밍하는 중 오류가 발생했습니다: {e}"
                    message_placeholder.error(full_response)

                response_text = full_response
                # --- 👆 여기까지 ---

                if sources:
                    source_info = "\n\n--- \n**참고 자료:**\n"
                    for source in sources:
                        page_number = source.get("page")
                        if page_number:
                            link = f"{PDF_URL}#page={page_number}"
                            source_info += f"- [{source['text']}]({link})\n"
                    st.markdown(source_info)
                    response_text += source_info

        st.session_state.messages.append({"role": "assistant", "content": response_text})