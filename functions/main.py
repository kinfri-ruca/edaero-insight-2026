import os
import json
import time
import re
import firebase_admin
from firebase_admin import credentials, storage, firestore
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

# Firebase Functions 라이브러리 임포트
from firebase_functions import storage_fn
from cloudevents.http import CloudEvent

# --- RUCAS LEE님 최종 설정 ---
VISION_MODEL = 'gemini-2.5-flash'
EXTRACTION_MODEL = 'gemini-2.5-flash'
# ------------------------------------

# 앱 초기화는 함수 밖에서 한번만 수행합니다.
# 클라우드 환경에서는 서비스 계정 키 파일이 필요 없습니다.
firebase_admin.initialize_app()
# Gemini API 키는 Cloud Function의 환경 변수로 설정할 것입니다.
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))


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
    
    return full_text

def get_common_info(full_text, db, pdf_filename):
    """(2-1단계) 전체 텍스트에서 공통 정보를 추출합니다."""
    update_progress(db, pdf_filename, "공통 정보 분석 중...", 50)
    model = genai.GenerativeModel(EXTRACTION_MODEL)
    prompt = f"주어진 입시요강 전체 텍스트에서 모든 지원자에게 공통적으로 적용되는 '공통 정보'를 찾아서 JSON 객체 형식으로 만들어줘. 찾아야 할 항목: \"application_period\", \"application_procedure\", \"application_fee\", \"csat_english_method\", \"csat_history_method\". 응답은 오직 JSON 객체만 포함해야 한다. --- 분석할 텍스트 ---\n{full_text}"
    try:
        response = model.generate_content(prompt, request_options={"timeout": 600})
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_text)
    except Exception as e:
        update_progress(db, pdf_filename, "오류: 공통 정보 분석 실패", 55)
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
        text_chunk_with_pages = "".join([f"\n\n--- Page {p_num + 1} ---\n{pages[p_num]}" for p_num in range(start_index, min(end_index, len(pages)))])
        
        model = genai.GenerativeModel(EXTRACTION_MODEL)
        prompt = f"주어진 입시요강 텍스트 일부를 분석하여, `department_info` JSON 배열 형식으로 만들어줘. 찾아야 할 항목: \"major\", \"recruitment_unit\", \"selection_category\", \"recruitment_number\", \"csat_ratios\", \"evaluation_method\", \"source_page\". 절대로 응답을 요약하거나 생략하지 말고, 찾은 모든 학과 정보를 생성해야 한다. 분석할 정보가 없다면 빈 배열 `[]`을 반환하세요. 응답은 오직 JSON 배열 형식이어야 한다. --- 분석할 텍스트 ---\n{text_chunk_with_pages}"
        try:
            response = model.generate_content(prompt, request_options={"timeout": 600})
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            chunk_result = json.loads(cleaned_text)
            final_department_info.extend(chunk_result)
        except Exception as e:
            print(f"    - ❗️ 청크 {i+1} 처리 중 오류 발생: {e}")
        time.sleep(1)
        
    return final_department_info


@storage_fn.on_object_finalized(region="asia-northeast3")
def process_pdf_on_upload(event: CloudEvent) -> None:
    """Storage에 PDF 파일이 업로드되면 자동으로 실행되는 메인 Cloud Function."""
    
    bucket_name = event.data["bucket"]
    file_name = event.data["name"]
    content_type = event.data["contentType"]

    if not content_type or not content_type.startswith("application/pdf"):
        print(f"'{file_name}'은 PDF 파일이 아니므로 무시합니다.")
        return

    db_client = firestore.client()
    storage_bucket = storage.bucket(bucket_name)

    update_progress(db_client, file_name, "PDF 다운로드 중...", 5)
    blob = storage_bucket.blob(file_name)
    if not blob:
        update_progress(db_client, file_name, "오류: Storage에서 파일을 찾을 수 없음", -1)
        return
    pdf_bytes = blob.download_as_bytes()
    
    full_text = extract_text_with_vision(pdf_bytes, db_client, file_name)
    
    common_info = get_common_info(full_text, db_client, file_name)
    department_info = structure_department_info_by_chunks(full_text, db_client, file_name)
    
    update_progress(db_client, file_name, "최종 JSON 파일 생성 중...", 95)
    final_json = {
        "university": "대학교 이름", "year": "2026", "document_title": file_name,
        "common_info": common_info, "department_info": department_info
    }
    
    # Cloud Function 환경에서는 /tmp/ 디렉토리에만 파일을 쓸 수 있습니다.
    output_filename = f"result_{os.path.splitext(file_name)[0]}_final.json"
    local_tmp_path = f"/tmp/{output_filename}"
    with open(local_tmp_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    
    # 최종 결과 파일을 다시 Storage의 'results' 폴더에 업로드
    result_blob = storage_bucket.blob(f"results/{output_filename}")
    result_blob.upload_from_filename(local_tmp_path)
    print(f"최종 결과 파일을 Storage 'results/' 폴더에 업로드했습니다.")
    
    update_progress(db_client, file_name, "완료", 100)