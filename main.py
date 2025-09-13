import os
import json
import time
import firebase_admin
from firebase_admin import credentials
from google.cloud import storage
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

# --- 설정 ---
SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'
BUCKET_NAME = 'edaero-insight-2026.firebasestorage.app' 
PDF_FILE_NAME = '2026_서울대학교_정시.pdf'
# ------------------------------------

# 1. API 키 및 Firebase 초기화
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini API 키가 성공적으로 설정되었습니다.")
except KeyError:
    print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요. (PowerShell: $env:GOOGLE_API_KEY='YOUR_KEY')")
    exit()

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)

def extract_text_from_image_with_gemini(image_data):
    """(1단계 함수) Gemini Vision을 사용하여 이미지에서 텍스트를 추출합니다."""
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = "이 이미지는 문서의 한 페이지입니다. 이 페이지에 보이는 모든 텍스트를 빠짐없이, 순서대로 정확하게 추출해주세요."

    try:
        response = model.generate_content([prompt, image_data], request_options={"timeout": 600})
        return response.text
    except Exception as e:
        print(f"  - ❗️ Gemini Vision API 호출 중 오류 발생: {e}")
        return f"Error processing page: {e}"

def structure_text_with_gemini(full_text, file_name):
    """(2단계 함수) 추출된 전체 텍스트를 바탕으로 Gemini를 이용해 JSON을 생성합니다."""
    print("\n🧠 Gemini API에 최종 JSON 구조화를 요청합니다...")
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    prompt = f"""
    당신은 대학 입시요강 분석 전문가입니다. 아래 텍스트는 '{file_name}' 파일의 전체 내용입니다.
    이 텍스트를 분석하여, 아래 JSON 스키마에 따라 모든 전형 정보를 추출하고 구조화된 JSON을 생성해주세요.
    존재하지 않는 정보는 null로 처리하고, 반드시 JSON 형식으로만 응답해주세요. 불필요한 설명이나 ```json 같은 마크다운은 포함하지 마세요.

    [
      {{
        "university": "대학교 이름", "year": "학년도", "admission_type": "수시모집 또는 정시모집",
        "selection_category": "전형명", "recruitment_unit": "모집 단위", "major": "학과/학부",
        "eligibility": "지원 자격 조건", "evaluation_method": "전형 방법",
        "csat_minimums": "수능 최저학력기준", "required_documents": ["제출 서류 목록"],
        "source_page": "정보가 있었던 원본 페이지 번호"
      }}
    ]

    --- 분석할 텍스트 ---
    {full_text}
    """

    try:
        response = model.generate_content(prompt, request_options={"timeout": 1200})
        print("✅ Gemini JSON 구조화 완료!")
        return response.text # 파싱은 main 함수에서 진행
    except Exception as e:
        print(f"❌ Gemini JSON 구조화 중 오류 발생: {e}")
        return None

def main():
    """메인 실행 함수"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(PDF_FILE_NAME)

    print(f"'{BUCKET_NAME}' 버킷에서 '{PDF_FILE_NAME}' 파일을 다운로드합니다...")
    pdf_bytes = blob.download_as_bytes()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_texts = []
    total_pages = pdf_document.page_count
    print(f"📄 총 {total_pages} 페이지의 시각적 분석을 시작합니다...")

    # --- 1단계: 각 페이지를 이미지로 변환하여 Gemini Vision으로 텍스트 추출 ---
    for page_num in range(total_pages):
        print(f"  - Vision API 처리 중: {page_num + 1} / {total_pages} 페이지...")
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text_from_page = extract_text_from_image_with_gemini(img)
        page_texts.append(f"--- Page {page_num + 1} ---\n{text_from_page}")
        time.sleep(1)

    full_extracted_text = "\n\n".join(page_texts)

    # --- 2단계: 추출된 전체 텍스트를 Gemini로 구조화 ---
    gemini_response_text = structure_text_with_gemini(full_extracted_text, PDF_FILE_NAME)

    if gemini_response_text:
        # --- 3단계: '선 저장, 후 처리' 로직 ---
        raw_filename = f"result_{os.path.splitext(PDF_FILE_NAME)[0]}_raw.txt"
        with open(raw_filename, 'w', encoding='utf-8') as f:
            f.write(gemini_response_text)
        print(f"\n📄 Gemini의 원본 응답을 '{raw_filename}'에 안전하게 저장했습니다.")

        try:
            cleaned_text = gemini_response_text.strip().replace("```json", "").replace("```", "")
            structured_data = json.loads(cleaned_text)

            output_filename = f"result_{os.path.splitext(PDF_FILE_NAME)[0]}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            print(f"✨ 최종 결과가 '{output_filename}' 파일로 저장되었습니다.")

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: Gemini가 생성한 텍스트가 완벽한 JSON이 아닙니다.")
            print(f"   - 오류 내용: {e}")
            print(f"   - 원본 내용은 방금 저장된 '{raw_filename}' 파일을 열어 확인해주세요.")

if __name__ == "__main__":
    main()