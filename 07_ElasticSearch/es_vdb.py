import os
import requests
import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder
from langsmith import traceable
from langsmith import wrappers
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
print(f"DEBUG - LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
print(f"DEBUG - OPENAI_API_KEY 존재 여부: {bool(os.getenv('OPENAI_API_KEY'))}")

ES_URL = "http://localhost:9200"
ES_INDEX = "es_docs"

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

wv_client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST,
    http_port=WEAVIATE_HTTP_PORT,
    grpc_host=WEAVIATE_HOST,
    grpc_port=WEAVIATE_GRPC_PORT,
    http_secure=False,
    grpc_secure=False,
)

print("🔄 임베딩 모델 로딩 시작...")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ 임베딩 모델 로딩 완료!")

print("🔄 리랭커 모델 로딩 시작...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("✅ 리랭커 모델 로딩 완료!")

# OpenAI Client Setup
openai_client = wrappers.wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

@traceable(name="LLM Call")
def openai_llm(prompt: str) -> str:
    print("🤖 [LLM] OpenAI API 호출 시작합니다...") # <- 추가
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    print("🤖 [LLM] OpenAI API 응답 수신 완료!") # <- 추가
    return response.choices[0].message.content


def es_search(query: str, k: int = 5) -> list[dict]:
    """BM25 + kNN 하이브리드 검색 with RRF (ES 8.9+).

    ES가 내부적으로 두 결과 목록을 RRF로 합산해 반환한다.
    RRF 점수 = Σ 1 / (rank_constant + rank_i)  (rank_constant=60 권장)
    """
    query_vector = embedder.encode(query).tolist()
    url = f"{ES_URL}/{ES_INDEX}/_search"
    payload = {
        "size": k,
        # ── BM25 키워드 검색 ──────────────────────────────
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"]
            }
        },
        # ── kNN 벡터 검색 ─────────────────────────────────
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": k * 10   # ANN 후보 수: 높을수록 정확하지만 느림
        },
        # ── RRF 점수 합산 ─────────────────────────────────
        "rank": {
            "rrf": {
                "window_size": k * 4,  # 각 결과 목록에서 고려할 최대 순위
                "rank_constant": 60    # 낮을수록 상위 문서 가중치 강화
            }
        }
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if resp.status_code != 200:
            print(f"❌ ES 검색 실패 원인: {resp.text}")
        resp.raise_for_status()

        docs = []
        for hit in resp.json().get("hits", {}).get("hits", []):
            src = hit["_source"]
            docs.append({
                "id":     f"es:{hit['_id']}",
                "text":   src.get("content", ""),
                "title":  src.get("title", ""),
                "source": "es_hybrid",
                "score":  hit["_score"]   # RRF 합산 점수
            })
        return docs

    except Exception as e:
        print(f"[Warning] Elasticsearch hybrid search failed: {e}")
        return []

def weaviate_search(query, k=5):
    try:
        collection = wv_client.collections.get("Document")
        response = collection.query.bm25(
            query=query,
            limit=k
        )
        docs = []
        for obj in response.objects:
            props = obj.properties
            docs.append({
                "id": f"wv:{obj.uuid}",
                "text": props.get("content", ""),
                "title": props.get("title", ""),
                "source": "weaviate",
                "score": obj.metadata.score if obj.metadata and obj.metadata.score is not None else 0
            })
        return docs
    except Exception as e:
        print(f"[Warning] Weaviate query failed: {e}")
        return []

def merge_dedup(docs):
    seen = set()
    merged = []
    for d in docs:
        key = (d["title"], d["text"][:200])
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    return merged

def rerank(query, docs, top_k=5):
    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)
    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)
    docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
    return docs[:top_k]

def build_context(docs):
    chunks = []
    for i, d in enumerate(docs, 1):
        chunks.append(f"[{i}] {d['title']}\n{d['text']}")
    return "\n\n".join(chunks)

@traceable(name="RAG Pipeline")
def rag_pipeline(query, llm):
    es_docs = []
    wv_docs = []
    
    # Try searching Elasticsearch, handle case if index or server is empty/not setup
    try:
        es_docs = es_search(query, k=5)
    except Exception as e:
        print(f"[Warning] Elasticsearch search failed or index does not exist: {e}")
        
    # Try searching Weaviate
    try:
        wv_docs = weaviate_search(query, k=5)
        print("Weaviate에서 검색된 문서는 : ", wv_docs)
    except Exception as e:
        print(f"[Warning] Weaviate search failed: {e}")

    merged = merge_dedup(es_docs + wv_docs)
    
    if not merged:
        print("[Info] No documents found in Elasticsearch or Weaviate. Using empty list for rerank.")
        top_docs = []
        context = "No relevant context found."
    else:
        top_docs = rerank(query, merged, top_k=4)
        context = build_context(top_docs)

    prompt = f"""
You are a helpful assistant.
Answer only using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}
"""
    return llm(prompt), top_docs

if __name__ == "__main__":
    # Test query
    print("현재 파일 시작")
    test_query = ["Elasticsearch가 뭐야", "What is Elasticsearch", "Weaviate 란?"]

    for q in test_query:
        print(f"Running RAG pipeline for query: '{q}'...")
        try:
            answer, retrieved_docs = rag_pipeline(q, openai_llm)
            print("\n=== Answer ===")
            print(answer)
            print("\n=== Retrieved Documents ===")
            for idx, doc in enumerate(retrieved_docs, 1):
                print(f"[{idx}] Source: {doc['source']}, Title: {doc['title']}, Score: {doc.get('rerank_score', 0):.4f}")
                print(f"    Content: {doc['text'][:150]}...")
        except Exception as e:
            print(f"An error occurred during execution: {e}")
    wv_client.close()
