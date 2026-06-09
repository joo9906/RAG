import weaviate

client = weaviate.connect_to_local()
collection = client.collections.get("DocumentCollection")

try:
    # 하이브리드 쿼리 수행
    response = collection.query.hybrid(
        query="임베딩 최적화 방법",  # 유저의 자연어 질문
        alpha=0.5,                 # 0.5 = 벡터 검색 50% + 전통 키워드(BM25) 검색 50% 융합
        limit=5                    # 최종 스코어가 가장 높은 상위 5개 결과 추출
    )
    
    # 결과 정제 및 출력
    print("🔍 검색 결과 출력:\n" + "="*40)
    for obj in response.objects:
        print(f"📄 제목: {obj.properties['title']}")
        print(f"📝 내용 요약: {obj.properties['content'][:60]}...")
        print(f"📊 매칭 점수: {obj.metadata.score}")
        print("-" * 40)

finally:
    client.close()