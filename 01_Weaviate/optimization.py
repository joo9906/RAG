import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

# Weaviate 서버 연결 (로컬 환경 기준)
client = weaviate.connect_to_local()

try:
    # 기존에 동일한 이름의 컬렉션이 있다면 삭제 (테스트 초기화용)
    if client.collections.exists("DocumentCollection"):
        client.collections.delete("DocumentCollection")
    
    # 5만 건 환경 맞춤형 최적화 컬렉션 생성
    client.collections.create(
        name="DocumentCollection",
        description="1500토큰 분량의 문서 5만 건에 최적화된 설정",
        
        # [최적화 1] 외부에서 벡터를 직접 주입할 것이므로 내부 벡터라이저는 비활성화
        vectorizer_config=Configure.Vectorizer.none(),
        
        # [최적화 2] HNSW 알고리즘 조작으로 정확도 극대화
        vector_index_config=Configure.VectorIndex.hnsw(
            max_connections=64,               # 노드당 링크 수를 64개로 늘려 정확도 향상 (기본값 32)
            ef_construction=256,             # 인덱스 생성 시 탐색 범위를 256으로 넓혀 고품질 지도 생성 (기본값 128)
            ef=128,                          # 검색 시 골목길 탐색 범위를 128로 설정하여 정밀 검색 (기본값 -1)
            distance_metric=VectorDistances.COSINE # OpenAI 임베딩에 표준인 코사인 유사도 사용
        ),
        
        # [최적화 3] 메타데이터 스키마 정의 (키워드 검색 및 필터링용)
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT)
        ]
    )
    print("✅ 최적화 컬렉션 생성 완료!")

finally:
    client.close()


