import numpy as np
import os
import json
from dotenv import load_dotenv
import requests
import weaviate

# Weaviate랑 연결 - 환경 변수 사용
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

print(f"🔗 Weaviate 연결: {WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}")

client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST,
    http_port=WEAVIATE_HTTP_PORT,
    grpc_host=WEAVIATE_HOST,
    grpc_port=WEAVIATE_GRPC_PORT,
    http_secure=False,
    grpc_secure=False,
    )

print(client.is_ready()) # True가 나오면 연결 성공입니다.

import weaviate
import weaviate.classes.config as wvc

# Class 생성 및 필드 설정
existing = client.collections.list_all()

if "FirstCollection" not in existing:
    first_collection = client.collections.create(
        name="FirstCollection",
        description="처음으로 생성한 컬렉션",
        vector_config=wvc.Configure.Vectors.self_provided(), # 직접 임베딩 할거기 때문에 self_provided()를 사용합니다.
        properties=[
            wvc.Property(name="input", data_type=wvc.config.DataType.TEXT),
            wvc.Property(name="output", data_type=wvc.config.DataType.TEXT),
        ],
    )
    print("✅ FirstCollection 컬렉션 생성 완료")
else:
    print("✅ FirstCollection 이미 존재. 생성 생략.")

# jsonl 파일 가공을 위한 함수
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

class Embedding():
    def __init__(self):
        self.api_key = API_KEY
        self.emb_model = EMB_MODEL
        self.emb_url = EMB_URL
    # 임베딩
    def embed(self, text: str) -> list:
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.emb_model, "input": text}

        res = requests.post(self.emb_url, headers=headers, json=payload)
        res.raise_for_status()

        result = res.json()["data"][0]["embedding"]

        return [float(v) for v in result]
        
emb = Embedding()

first_data = load_jsonl("first_data.json")
first_collection = client.collections.get("FirstCollection")

for idx, d in enumerate(single_data):
    try:
        # 1. 임베딩 생성 (에러 처리 포함)
        vector_embedding = emb.embed(d["input"])

        # 2. 벡터 검증
        if not vector_embedding or len(vector_embedding) == 0:
            print(f"❌ Row {idx}: 벡터가 비어있습니다")
            continue

        # 3. 데이터 객체 생성
        data_object = {
            "input": d["input"].strip(),
            "output": d["output"].strip()
        }

        # 4. Insert 및 반환값 확인
        uuid = single_collection.data.insert(
            properties=data_object,
            vector=vector_embedding
        )
        print(f"✅ Single {idx}: 저장 완료 (UUID: {uuid})")

    except Exception as e:
        print(f"❌ Row {idx} 에러: {str(e)}")
        continue