import weaviate

client = weaviate.connect_to_local()
collection = client.collections.get("DocumentCollection")

# [샘플 데이터 준비] 실제 환경에서는 파싱/청킹된 데이터와 OpenAI 가 미리 계산한 벡터 리스트가 들어갑니다.
mock_documents = [
    {
        "title": f"프로젝트 최적화 가이드 문서 {i}",
        "content": f"이것은 약 1500토큰 분량의 긴 본문 텍스트 데이터입니다... 데이터 번호: {i}",
        "vector": [0.015] * 1536  # 실제 text-embedding-3-small에서 추출한 1536차원 부동소수점 리스트
    } for i in range(100) # 예시용 100건 데이터 생성
]

try:
    # fixed_size 배치를 활용하여 네트워크 오버헤드 최소화 및 안정적 삽입
    with collection.batch.fixed_size(batch_size=100) as batch:
        for doc in mock_documents:
            batch.add_object(
                properties={
                    "title": doc["title"],
                    "content": doc["content"]
                },
                vector=doc["vector"]  # 외부에서 가공 완료된 벡터를 직접 매핑
            )
            
    # 에러 모니터링
    if collection.batch.failed_objects:
        print(f"❌ 일부 데이터 삽입 실패: {len(collection.batch.failed_objects)}건")
    else:
        print(f"🚀 {len(mock_documents)}건의 데이터가 배치 모드로 완벽하게 삽입되었습니다!")

finally:
    client.close()