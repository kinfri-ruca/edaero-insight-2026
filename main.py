import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# 서비스 계정 키 파일 경로 설정
cred = credentials.Certificate('serviceAccountKey.json')

# Firebase 앱 초기화
firebase_admin.initialize_app(cred, {
    'projectId': 'edaero-insight-2026', # 이 부분을 실제 프로젝트 ID로 수정하세요!
})

# Firestore 클라이언트 가져오기
db = firestore.client()

print("✅ Firebase Admin SDK가 성공적으로 초기화되었습니다.")
print(f"Firestore에 연결되었습니다. 프로젝트 ID: {db.project}")