# admin.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
import time
import json 

# --- 최종 설정 ---
#SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'
BUCKET_NAME = 'edaero-insight-2026.firebasestorage.app' # RUCAS LEE님의 설정으로 수정했습니다.
# ------------------------------------

# Firebase 초기화
try:
    if not firebase_admin._apps:
        # Streamlit의 Secrets에서 서비스 계정 정보(JSON 텍스트)를 직접 읽어옵니다.
        service_account_info_str = st.secrets["firebase_service_account"]
        
        # 문자열을 Python 딕셔너리로 변환
        service_account_info = json.loads(service_account_info_str)
        
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
    
    db = firestore.client()

except Exception as e:
    st.error(f"Firebase 초기화 중 오류가 발생했습니다: {e}")
    st.stop()

st.set_page_config(page_title="edaeroAI Admin", layout="centered")
st.title("👨‍💻 edaeroAI 데이터 관리자")
st.markdown("---")

st.subheader("1. 입시요강 PDF 업로드")

uploaded_file = st.file_uploader("분석할 PDF 파일을 선택하세요.", type="pdf")

if uploaded_file is not None:
    st.success(f"'{uploaded_file.name}' 파일이 로드되었습니다.")
    
    if st.button("🚀 데이터 처리 시작"):
        # 1. 파일 업로드
        with st.spinner(f"'{uploaded_file.name}' 파일을 Firebase Storage에 업로드하는 중..."):
            try:
                bucket = storage.bucket()
                blob = bucket.blob(uploaded_file.name)
                # Streamlit의 UploadedFile 객체를 바이트로 읽어서 업로드
                uploaded_file.seek(0)
                blob.upload_from_file(uploaded_file, content_type='application/pdf')
                st.success("✅ 파일 업로드 성공! Cloud Function이 데이터 처리를 시작할 것입니다 (다음 단계에서 구현).")
            except Exception as e:
                st.error(f"파일 업로드 중 오류가 발생했습니다: {e}")
                st.stop()

        st.markdown("---")
        st.subheader("2. 데이터 처리 현황")

        # 2. Firestore에서 진행 상황 실시간 모니터링
        progress_bar = st.progress(0, text="작업 대기 중...")
        
        doc_ref = db.collection('progress').document(uploaded_file.name)

        # 문서가 생성될 때까지 잠시 대기
        time.sleep(5) 

        while True:
            doc = doc_ref.get()
            if doc.exists:
                progress_data = doc.to_dict()
                progress = progress_data.get('progress', 0)
                status = progress_data.get('status', '상태 정보 없음')
                
                if progress >= 0:
                    progress_bar.progress(progress, text=status)
                else: 
                    progress_bar.progress(100, text="오류 발생")
                    st.error(status)
                    break

                if progress >= 100:
                    st.success("🎉 모든 작업이 성공적으로 완료되었습니다!")
                    break
            else:
                st.info("데이터 처리 시작을 기다리는 중입니다...")
            
            time.sleep(2)