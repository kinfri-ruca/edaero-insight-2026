# check_storage.py
import firebase_admin
from firebase_admin import credentials, storage

SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'
BUCKET_NAME = 'edaero-insight-2026.firebasestorage.app'

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})

bucket = storage.bucket()
print(f"✅ '{BUCKET_NAME}' 버킷에 성공적으로 연결되었습니다.")
print("--- 버킷 안의 파일 목록 ---")
files = bucket.list_blobs()
found = False
for file in files:
    print(f"- {file.name}")
    found = True

if not found:
    print("버킷이 비어있습니다.")
print("--------------------------")