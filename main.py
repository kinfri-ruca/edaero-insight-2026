import os
import json
import time
import re
import firebase_admin
from firebase_admin import credentials, storage, firestore
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

# --- 설정 ---
SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'
BUCKET_NAME = 'edaero-insight-2026.firebasestorage.app'
VISION_MODEL = 'gemini-2.5-flash'
EXTRACTION_MODEL = 'gemini-2.5-flash' 
# ------------------------------------

def initialize_services():
    """API 키와 Firebase 서비스를 초기화하고 클라이언트 객체들을 반환합니다."""
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
        print("✅ Gemini API 키가 성공적으로 설정되었습니다.")
    except KeyError:
        print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요.")
        return None, None

    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
            firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
            print("✅ Firebase Admin SDK가 성공적으로 초기화되었습니다.")
        except Exception as e:
            print(f"❌ Firebase 초기화 중 오류 발생: {e}")
            return None, None
            
    try:
        db_client = firestore.client()
        storage_bucket = storage.bucket()
        print("✅ Firestore 및 Storage 클라이언트 연결 성공.")
        return db_client, storage_bucket
    except Exception as e:
        print(f"❌ Firestore 또는 Storage 클라이언트 연결 중 오류 발생: {e}")
        return None, None

def update_progress(db, filename, status, progress):
    """Firestore에 현재 진행 상태를 업데이트합니다."""
    if db:
        doc_ref = db.collection('progress').document(filename)
        doc_ref.set({'status': status, 'progress': progress, 'timestamp': firestore.SERVER_TIMESTAMP}, merge=True)

def extract_text_with_vision(pdf_bytes, db, pdf_filename):
    """(1단계) Vision API 텍스트 추출 및 진행 상황 보고"""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = pdf_document.page_count
    full_text = ""
    base_progress, progress_range = 10, 40

    for page_num in range(total_pages):
        current_progress = base_progress + int(((page_num + 1) / total_pages) * progress_range)
        update_progress(db, pdf_filename, f"텍스트 추출 중 ({page_num + 1}/{total_pages} 페이지)", current_progress)
        
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        model = genai.GenerativeModel(VISION_MODEL)
        prompt = "이 이미지는 문서의 한 페이지입니다. 이 페이지에 보이는 모든 텍스트를 빠짐없이, 순서대로 정확하게 추출해주세요."
        
        try:
            response = model.generate_content([prompt, img], request_options={"timeout": 600})
            full_text += f"\n\n--- Page {page_num + 1} ---\n{response.text}"
            time.sleep(1)
        except Exception as e:
            update_progress(db, pdf_filename, f"오류: {page_num + 1} 페이지 처리 실패", current_progress)
            full_text += f"\n\n--- Page {page_num + 1} ---\nError processing page: {e}"

    raw_text_filename = f"result_{os.path.splitext(pdf_filename)[0]}_raw_text.txt"
    with open(raw_text_filename, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    return full_text

def get_common_info(full_text, db, pdf_filename):
    """(2-1단계) 전체 텍스트에서 공통 정보를 추출합니다."""
    update_progress(db, pdf_filename, "공통 정보 분석 중...", 50)
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
        return json.loads(cleaned_text)
    except Exception as e:
        update_progress(db, pdf_filename, "오류: 공통 정보 분석 실패", 55)
        print(f"❌ 'common_info' 정보 추출 중 오류 발생: {e}")
        return {}

def structure_department_info_by_chunks(full_text, db, pdf_filename):
    """(2-2단계) 전체 텍스트를 청크로 나누어 학과별 정보를 추출하고 종합합니다."""
    update_progress(db, pdf_filename, "학과별 정보 상세 분석 준비 중...", 70)
    pages = re.split(r'--- Page \d+ ---', full_text)
    pages = [p.strip() for p in pages if p.strip()]
    
    final_department_info = []
    chunk_size = 10
    num_chunks = (len(pages) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        current_progress = 70 + int(((i + 1) / num_chunks) * 25)
        update_progress(db, pdf_filename, f"학과별 정보 분석 중 (청크 {i+1}/{num_chunks})", current_progress)

        start_index = i * chunk_size
        end_index = start_index + chunk_size
        text_chunk_with_pages = ""
        for page_num in range(start_index, min(end_index, len(pages))):
            text_chunk_with_pages += f"\n\n--- Page {page_num + 1} ---\n{pages[page_num]}"
        
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
        except Exception as e:
            print(f"    - ❗️ 청크 {i+1} 처리 중 오류 발생: {e}")
        time.sleep(1)
        
    return final_department_info

def main(db, bucket, pdf_filename):
    """메인 실행 함수"""
    print("--- [디버깅] main 함수 시작 ---")
    
    update_progress(db, pdf_filename, "PDF 다운로드 중...", 5)
    blob = bucket.blob(pdf_filename)
    if not blob.exists():
        print(f"--- [디버깅] 오류: Storage에서 '{pdf_filename}' 파일을 찾을 수 없습니다. ---")
        update_progress(db, pdf_filename, "오류: 파일을 찾을 수 없음", -1)
        return
    pdf_bytes = blob.download_as_bytes()
    print("--- [디버깅] PDF 다운로드 완료 ---")
    
    full_text = extract_text_with_vision(pdf_bytes, db, pdf_filename)
    print(f"--- [디버깅] 텍스트 추출 완료. 총 글자 수: {len(full_text)} ---")
    
    # 텍스트 추출이 실패했는지 확인
    if len(full_text) < 100: # 텍스트가 너무 짧으면 문제가 있는 것으로 간주
        print("--- [디버깅] 오류: 추출된 텍스트가 너무 짧습니다. 프로세스를 중단합니다. ---")
        update_progress(db, pdf_filename, "오류: 텍스트 추출 실패", -1)
        return

    common_info = get_common_info(full_text, db, pdf_filename)
    print(f"--- [디버깅] 공통 정보 추출 완료: {common_info} ---")
    
    department_info = structure_department_info_by_chunks(full_text, db, pdf_filename)
    print(f"--- [디버깅] 학과별 정보 추출 완료. 총 {len(department_info)}개 학과 발견 ---")
    
    update_progress(db, pdf_filename, "최종 JSON 파일 생성 중...", 95)
    final_json = {
        "university": "대학교 이름",
        "year": "2026",
        "document_title": pdf_filename,
        "common_info": common_info,
        "department_info": department_info
    }
    
    output_filename = f"result_{os.path.splitext(pdf_filename)[0]}_final.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    
    print("--- [디버깅] JSON 파일 저장 완료 ---")
    update_progress(db, pdf_filename, "완료", 100)
    print(f"✨ 최종 통합 JSON 생성 완료! 결과가 '{output_filename}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    db_client, storage_bucket = initialize_services()
    if db_client and storage_bucket:
        test_pdf_file = "2026_서울시립대학교_정시.pdf" 
        main(db_client, storage_bucket, test_pdf_file)