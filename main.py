import os
import json
import time
import re
import firebase_admin
from firebase_admin import credentials
from google.cloud import storage
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

# --- 설정 ---
SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'
BUCKET_NAME = 'edaero-insight-2026.firebasestorage.app' 
PDF_FILE_NAME = '2026_서울시립대학교_정시.pdf'

# --- 결과 파일 이름 설정 ---
RAW_TEXT_FILENAME = f"result_{os.path.splitext(PDF_FILE_NAME)[0]}_raw_text.txt"
FINAL_JSON_FILENAME = f"result_{os.path.splitext(PDF_FILE_NAME)[0]}_final.json"

# --- 모델 설정 ---
# 품질 우선: gemini-1.5-pro-latest 또는 gemini-2.5-pro 등
# 속도 우선: gemini-1.5-flash-latest 또는 gemini-2.5-flash 등
VISION_MODEL = 'gemini-2.5-flash'
EXTRACTION_MODEL = 'gemini-2.5-flash' 
# ------------------------------------

def initialize_services():
    """API 키와 Firebase 서비스를 초기화합니다."""
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
        print("✅ Gemini API 키가 성공적으로 설정되었습니다.")
    except KeyError:
        print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요.")
        return False

    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)
    return True

def extract_text_with_vision(pdf_bytes):
    """(1단계) Vision API를 사용하여 PDF의 모든 페이지에서 고품질 텍스트를 추출합니다."""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = pdf_document.page_count
    print(f"📄 총 {total_pages} 페이지의 시각적 분석(Vision API)을 시작합니다...")
    
    full_text = ""
    for page_num in range(total_pages):
        print(f"  - Vision API 처리 중: {page_num + 1} / {total_pages} 페이지...")
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=300) # 고해상도 DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        model = genai.GenerativeModel(VISION_MODEL)
        prompt = "이 이미지는 문서의 한 페이지입니다. 이 페이지에 보이는 모든 텍스트를 빠짐없이, 순서대로 정확하게 추출해주세요."
        
        try:
            response = model.generate_content([prompt, img], request_options={"timeout": 600})
            full_text += f"\n\n--- Page {page_num + 1} ---\n{response.text}"
            time.sleep(1)
        except Exception as e:
            print(f"  - ❗️ {page_num + 1} 페이지 처리 중 오류 발생: {e}")
            full_text += f"\n\n--- Page {page_num + 1} ---\nError processing page."

    print("✅ 모든 페이지 텍스트 추출 완료.")
    with open(RAW_TEXT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"📄 추출된 전체 텍스트를 '{RAW_TEXT_FILENAME}'에 안전하게 저장했습니다.")
    return full_text

def get_common_info(full_text):
    """(2-1단계) 전체 텍스트에서 공통 정보를 추출합니다."""
    print("\n🧠 Gemini에 'common_info' 추출을 요청합니다...")
    model = genai.GenerativeModel(EXTRACTION_MODEL)
    prompt = f"""
    주어진 입시요강 전체 텍스트에서 모든 지원자에게 공통적으로 적용되는 '공통 정보'를 찾아서 JSON 객체 형식으로 만들어줘.
    찾아야 할 항목: "application_period", "application_procedure", "application_fee", "csat_english_method", "csat_history_method"
    응답은 오직 JSON 객체만 포함해야 한다.
    --- 분석할 텍스트 ---
    {full_text}
    """
    try:
        response = model.generate_content(prompt, request_options={"timeout": 600})
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        print("✅ 'common_info' 추출 완료.")
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"❌ 'common_info' 정보 추출 중 오류 발생: {e}")
        return {} # 실패 시 빈 객체 반환

def structure_department_info_by_chunks(full_text):
    """(2-2단계) 전체 텍스트를 청크로 나누어 학과별 정보를 추출하고 종합합니다."""
    print("\n🧠 '분할 정복' 방식으로 'department_info' 추출을 시작합니다...")
    
    pages = re.split(r'--- Page \d+ ---', full_text)
    pages = [p.strip() for p in pages if p.strip()]
    
    final_department_info = []
    chunk_size = 10
    num_chunks = (len(pages) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        text_chunk_with_pages = ""
        for page_num in range(start_index, min(end_index, len(pages))):
            text_chunk_with_pages += f"\n\n--- Page {page_num + 1} ---\n{pages[page_num]}"

        print(f"  - 청크 {i+1}/{num_chunks} 처리 중...")
        model = genai.GenerativeModel(EXTRACTION_MODEL)
        prompt = f"""
        주어진 입시요강 텍스트 일부를 분석하여, `department_info` JSON 배열 형식으로 만들어줘.
        찾아야 할 항목: "major", "recruitment_unit", "selection_category", "recruitment_number", "csat_ratios", "evaluation_method", "source_page".
        절대로 응답을 요약하거나 생략하지 말고, 찾은 모든 학과 정보를 생성해야 한다. 분석할 정보가 없다면 빈 배열 `[]`을 반환하세요.
        응답은 오직 JSON 배열 형식이어야 한다.
        --- 분석할 텍스트 ---
        {text_chunk_with_pages}
        """
        
        try:
            response = model.generate_content(prompt, request_options={"timeout": 600})
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            chunk_result = json.loads(cleaned_text)
            final_department_info.extend(chunk_result)
            print(f"    ✅ 청크 처리 완료. {len(chunk_result)}개의 항목 추가됨.")
        except Exception as e:
            print(f"    - ❗️ 청크 {i+1} 처리 중 오류 발생: {e}")
        time.sleep(1)
        
    return final_department_info

def main():
    """메인 실행 함수"""
    if not initialize_services(): return

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(PDF_FILE_NAME)
    print(f"'{BUCKET_NAME}' 버킷에서 '{PDF_FILE_NAME}' 파일을 다운로드합니다...")
    pdf_bytes = blob.download_as_bytes()
    
    # 1단계: Vision API로 고품질 텍스트 추출
    full_text = extract_text_with_vision(pdf_bytes)
    
    # 2단계: 공통 정보 및 학과별 정보 각각 추출
    common_info = get_common_info(full_text)
    department_info = structure_department_info_by_chunks(full_text)
    
    # 3단계: 최종 JSON 조립 및 저장
    final_json = {
        "university": "서울시립대학교",
        "year": "2026",
        "document_title": PDF_FILE_NAME,
        "common_info": common_info,
        "department_info": department_info
    }
    
    with open(FINAL_JSON_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    print(f"\n✨ 최종 통합 JSON 생성 완료! 결과가 '{FINAL_JSON_FILENAME}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()