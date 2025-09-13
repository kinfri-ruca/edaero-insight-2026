import os
import json
import time
import google.generativeai as genai

# --- 설정 ---
RAW_TEXT_FILE = 'result_2026_서울대학교_정시_raw.txt'
FINAL_JSON_FILE = 'result_2026_서울대학교_정시.json'
# ------------------------------------

# 1. API 키 설정
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini API 키가 성공적으로 설정되었습니다.")
except KeyError:
    print("❌ 에러: GOOGLE_API_KEY 환경 변수를 설정해주세요.")
    exit()

def structure_chunk_with_gemini(text_chunk):
    """텍스트 조각을 바탕으로 Gemini를 이용해 JSON을 생성합니다."""
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    prompt = f"""
    당신은 대학 입시요강 분석 전문가입니다. 아래 텍스트는 입시요강의 일부 내용입니다.
    이 텍스트를 분석하여, 아래 JSON 스키마에 따라 모든 전형 정보를 추출하고 구조화된 JSON 배열을 생성해주세요.
    만약 분석할 정보가 없다면 빈 배열 `[]`을 반환하세요.
    반드시 JSON 형식으로만 응답해주세요. 불필요한 설명이나 ```json 같은 마크다운은 포함하지 마세요.

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
    {text_chunk}
    """

    try:
        response = model.generate_content(prompt, request_options={"timeout": 600})
        return json.loads(response.text)
    except Exception as e:
        print(f"  - ❗️ Gemini 청크 처리 중 오류 발생: {e}")
        return [] # 오류 발생 시 빈 리스트 반환

def main():
    """메인 실행 함수"""
    print(f"'{RAW_TEXT_FILE}' 파일을 읽어옵니다...")
    with open(RAW_TEXT_FILE, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # 페이지 단위로 텍스트 분리
    pages = full_text.split('--- Page ')
    pages = [p for p in pages if p.strip()] # 빈 페이지 제거

    final_results = []

    # 10 페이지씩 묶어서 처리 (API 호출 횟수 조절)
    chunk_size = 10 
    num_chunks = (len(pages) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        text_chunk = "--- Page ".join(pages[start_index:end_index])

        print(f"\n🧠 청크 {i+1}/{num_chunks} JSON 구조화를 요청합니다...")

        structured_chunk = structure_chunk_with_gemini(text_chunk)

        if structured_chunk:
            final_results.extend(structured_chunk)
            print(f"  - ✅ 청크 처리 완료. {len(structured_chunk)}개의 항목 추가됨.")
        else:
            print(f"  - ⚠️ 해당 청크에서 유효한 정보를 찾지 못했습니다.")

        time.sleep(1) # API 과부하 방지

    # 최종 결과를 .json 파일로 저장
    with open(FINAL_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n✨ 최종 정제 완료! 총 {len(final_results)}개의 항목이 '{FINAL_JSON_FILE}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()