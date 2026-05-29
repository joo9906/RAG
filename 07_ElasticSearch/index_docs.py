"""
index_docs.py
docs/ 폴더의 .txt 파일들을 Elasticsearch와 Weaviate에 인덱싱하는 스크립트
ES: BM25(Nori) + dense_vector(kNN) 하이브리드 인덱스
"""
import os
import requests
import weaviate
import weaviate.classes.config as wvc
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

ES_URL = "http://localhost:9200"
ES_INDEX = "es_docs"
DOCS_DIR = Path(__file__).parent / "docs"

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384차원, 한/영 모두 지원
EMBED_DIMS = 384


# ──────────────────────────────────────────
# 문서 로드
# ──────────────────────────────────────────
def load_docs(docs_dir: Path) -> list[dict]:
    docs = []
    for txt_file in sorted(docs_dir.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8").strip()
        title = txt_file.stem.replace("_", " ").title()
        docs.append({"title": title, "content": content, "filename": txt_file.name})
    return docs


# ──────────────────────────────────────────
# Elasticsearch 인덱싱
# ──────────────────────────────────────────
def setup_es_index():
    """docs 인덱스가 없으면 생성하고, 있으면 삭제 후 재생성"""
    resp = requests.head(f"{ES_URL}/{ES_INDEX}")
    if resp.status_code == 200:
        print(f"[ES] '{ES_INDEX}' 인덱스가 이미 존재합니다. 삭제 후 재생성합니다.")
        requests.delete(f"{ES_URL}/{ES_INDEX}")

    nori_settings = {
        "settings": {
            "index": {
                "analysis": {
                    "analyzer": {
                        "nori_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer"
                        }
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "nori_analyzer",
                    "search_analyzer": "nori_analyzer"
                },
                "content": {
                    "type": "text",
                    "analyzer": "nori_analyzer",
                    "search_analyzer": "nori_analyzer"
                },
                "status": {
                    "type": "keyword"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBED_DIMS,
                    "index": True,          # kNN ANN 인덱스 활성화
                    "similarity": "cosine"  # 코사인 유사도 사용
                }
            }
        }
    }

    resp = requests.put(
        f"{ES_URL}/{ES_INDEX}",
        json=nori_settings,
        headers={"Content-Type": "application/json"}
    )
    
    # 에러가 나면 어떤 에러인지 상세 내용을 찍어주도록 print 추가
    if resp.status_code != 200:
        print(f"❌ 생성 실패 원인 메시지: {resp.text}")
        
    resp.raise_for_status()
    print(f"[ES] '{ES_INDEX}' 인덱스 생성 완료")


def index_to_es(docs: list[dict]):
    setup_es_index()

    print(f"[ES] 임베딩 모델 로딩: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    texts = [doc["content"] for doc in docs]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    print(f"[ES] 임베딩 생성 완료: {len(embeddings)}개 ({EMBED_DIMS}차원)\n")

    for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
        payload = {
            "title":     doc["title"],
            "content":   doc["content"],
            "status":    "active",
            "embedding": embedding.tolist()
        }
        resp = requests.post(
            f"{ES_URL}/{ES_INDEX}/_doc/{i+1}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        print(f"[ES] 문서 삽입 완료: [{i+1}] {doc['title']}")

    requests.post(f"{ES_URL}/{ES_INDEX}/_refresh")
    print("[ES] 인덱싱 완료\n")


# ──────────────────────────────────────────
# Weaviate 인덱싱
# ──────────────────────────────────────────
def index_to_weaviate(docs: list[dict]):
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_HTTP_PORT,
        grpc_host=WEAVIATE_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        http_secure=False,
        grpc_secure=False,
    )

    try:
        existing = client.collections.list_all()
        if "Document" in existing:
            print("[Weaviate] 'Document' 컬렉션이 이미 존재합니다. 삭제 후 재생성합니다.")
            client.collections.delete("Document")

        # 벡터 없이 BM25 키워드 검색만 사용
        collection = client.collections.create(
            name="Document",
            description="RAG 테스트용 문서 컬렉션",
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            properties=[
                wvc.Property(name="title",   data_type=wvc.DataType.TEXT),
                wvc.Property(name="content", data_type=wvc.DataType.TEXT),
            ],
        )
        print("[Weaviate] 'Document' 컬렉션 생성 완료")

        for doc in docs:
            collection.data.insert({"title": doc["title"], "content": doc["content"]})
            print(f"[Weaviate] 문서 삽입 완료: {doc['title']}")

        print("[Weaviate] 인덱싱 완료\n")

    finally:
        client.close()


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == "__main__":
    docs = load_docs(DOCS_DIR)
    if not docs:
        print(f"[Error] {DOCS_DIR} 폴더에 .txt 파일이 없습니다.")
        exit(1)

    print(f"로드된 문서 {len(docs)}개: {[d['title'] for d in docs]}\n")

    print("=== Elasticsearch 인덱싱 ===")
    index_to_es(docs)

    print("=== Weaviate 인덱싱 ===")
    index_to_weaviate(docs)

    print("✅ 모든 인덱싱 완료! 이제 es_vdb.py를 실행하세요.")
