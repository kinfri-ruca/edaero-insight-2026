import os
import firebase_admin
from firebase_admin import credentials
from google.cloud import storage
import fitz  # PyMuPDF
from PIL import Image
import io
import google.generativeai as genai

# --- 설정 (이 부분을 수정해주세요) ---
SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'
# Firebase Storage URL (gs:// 다음 부분)
BUCKET_NAME = 'edaero-insight-2026.firebasestorage.app' 
# Storage에 업로드한 PDF 파일 이름
PDF_FILE_NAME = '2026_서울대학교_정시.pdf'
# ------------------------------------

# 1. API 키 설정 (환경 변수에서 가져오기)
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini API 키가 성공적으로 설정되었습니다.")
except KeyError:
    print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요.")
    exit()

# 2. Firebase Admin SDK 초기화
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)

def analyze_image_with_gemini(image_data, prompt):
    """Gemini Vision API를 사용하여 이미지 내용을 분석합니다."""
    print("\n🧠 Gemini Vision API에 분석을 요청합니다...")
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    try:
        response = model.generate_content([prompt, image_data])
        print("✅ Gemini 분석 완료!")
        return response.text
    except Exception as e:
        print(f"❌ Gemini API 호출 중 오류 발생: {e}")
        return None

def process_pdf_from_storage(bucket_name, file_name):
    """Storage에서 PDF를 로드하고 첫 페이지를 Gemini로 분석합니다."""
    print(f"'{bucket_name}' 버킷에서 '{file_name}' 파일을 다운로드합니다...")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        pdf_bytes = blob.download_as_bytes()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        print(f"✅ 성공: '{file_name}' 파일을 열었습니다. (총 {pdf_document.page_count} 페이지)")

        if pdf_document.page_count > 0:
            # 첫 페이지를 고해상도 이미지로 변환
            page = pdf_document.load_page(0)
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Gemini에게 전달할 프롬프트
            prompt = "이 이미지는 대학 입시요강의 한 페이지입니다. 이 페이지에 있는 모든 텍스트를 추출하고, '전형명', '모집단위', '지원자격'과 같은 핵심 정보를 요약해주세요."

            # Gemini 분석 실행
            analysis_result = analyze_image_with_gemini(img, prompt)

            if analysis_result:
                print("\n--- [Gemini 분석 결과 (첫 페이지)] ---")
                print(analysis_result)
                print("------------------------------------")

        pdf_document.close()
        return True

    except Exception as e:
        print(f"❌ 파일 처리 중 오류 발생: {e}")
        return False

# 메인 코드 실행
if __name__ == "__main__":
    process_pdf_from_storage(BUCKET_NAME, PDF_FILE_NAME)