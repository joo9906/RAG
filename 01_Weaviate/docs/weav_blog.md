# [Weaviate] 프로독션 레벨 벡터 DB 최적화 가이드: HNSW, 이진 양자화(BQ), 멀티 타겟 검색, 그리고 숨겨진 고도화 팁까지 (v4 완전 정복)

안녕하세요! 현재 제가 재직중인 회사의 VDB로 Weaviate를 사용하는 만큼, 많은 최적화/고도화 방법을 공부하고 있습니다. 특히 RAG 기반의 멀티 에이전트 최적화가 중요한만큼 더 깊은 내용을 적어보겠습니다.

서비스 초기 단계에서는 기본 설정으로도 잘 돌아가지만, 데이터가 수만 ~ 수백만 건으로 늘어나면 **검색 속도 저하(Latency 지연), 메모리 비용, 그리고 OOM(Out Of Memory) 서버 다운**이라는 삼중고를 겪게 됩니다.

이번 글에서는 **Weaviate Python Client v4**를 기준으로, 아키텍처 고도화 이론과 코드까지 파헤쳐 보겠습니다.

---

## 1. 벡터 인덱스(Vector Index)란?

저는 처음에 벡터 인덱스가 정확히 뭔지 몰랐습니다. *"벡터 인덱스도 결국 데이터에 붙는 메타데이터의 일종이 아닐까?"* 했어요. 하지만 공부해보니 두 개념은 수학적 구조와 목적이 완전히 다르더군요.

- **메타데이터(Metadata):** 데이터에 붙는 정형화된 '이름표'입니다. 파일 크기, 카테고리 등 텍스트/숫자로 이루어지며, 전통적인 관계형 DB처럼 **'정확한 키워드 일치'나 '조건 필터링'** 에 쓰입니다. 아래 예시가 메타데이터라고 보면 됩니다.

```python
example = {
    "촬영 일시": "2026-05-15 14:30:00",
    "카메라 기종": "iPhone 16 Pro",
    "위치 정보": "위도 37.382, 경도 126.655",
    "해상도": "4032 x 3024",
    "파일 크기": "3.2 MB"
}
```

- **벡터 인덱스(Vector Index):** AI 모델이 고차원 공간에 찍어놓은 **데이터의 '수학적 좌표 지도'** 입니다. 데이터의 내재된 의미와 맥락을 거리로 계산하여 **'의미적 유사도 검색'** 을 수행합니다. 아래 사진도 AI로 만들다 보니 뭔가 이해가 어려운데, 그냥 x, y, z 좌표 상에 점 하나 찍어둔거 생각하면 됩니다.

| 구분 | 메타데이터 (Metadata) | 벡터 인덱스 (Vector Index) |
| :--- | :--- | :--- |
| **작동 방식** | 이진 비교 및 인덱스 매칭 | 고차원 공간 내 거리 계산 (Cosine, L2 등) |
| **아키텍처 역할** | 검색 범위를 좁히는 **'필터'** | 맥락적 정답을 찾아가는 **'네비게이션'** |
| **주요 인덱스 알고리즘** | B-Tree, 역색인 (Inverted Index) | HNSW, IVF, Flat |

---

## 2. HNSW 인덱스 튜닝

Weaviate는 대규모 데이터셋에서 초고속 근사 최근접 이웃(ANN) 검색을 구현하기 위해 **HNSW(Hierarchical Navigable Small World)** 알고리즘을 기본 인덱스로 사용합니다. HNSW는 점들을 계층형 그래프 구조로 연결합니다. 지도 앱에서 '고속도로(상위 레이어)'를 타고 크게 도약한 뒤, 목적지 근처에서 '일반 도로(하위 레이어)'로 내려와 정밀하게 길을 찾는 내비게이션 메커니즘입니다. 아래 그림을 참고하면 이해가 더 쉽게 되실겁니다.

이 그래프의 촘촘함을 조작하는 핵심 파라미터 3가지는 다음과 같습니다. 이 파라미터들은 실제 weaviate 내에서 collection create 시에 사용되는 인자들입니다.

#### ① `maxConnections`

각 노드가 가질 수 있는 최대 도로 수입니다. 값이 클수록 지도가 촘촘해져 검색 정확도가 올라가지만, 인덱스가 차지하는 RAM 용량이 커집니다. 문서 5만개(75만 토큰) 기준 약 2GB 정도 나오더라구요

#### ② `efConstruction`

인덱스를 처음 빌드할 때 주변을 얼마나 샅샅이 뒤질지 결정합니다. 값이 클수록 초기 데이터 적재 속도는 느려지지만 고품질 도로망이 완성됩니다.

#### ③ `ef`

검색(쿼리) 시점에 골목길을 얼마나 돌아다닐지 지정합니다. 정확도를 높이려면 상향하고, 속도를 극대화하려면 낮춥니다.

> 💡 **튜닝 팁:** `maxConnections`와 `efConstruction`은 인덱스 생성 시점에 결정되며 사후 변경 비용이 큽니다. 반면 `ef`는 쿼리마다 조정 가능하므로, 운영 중 A/B 테스트로 적정값을 찾기 좋습니다.

### 💻 예시 코드 1: 고정밀 HNSW 단일 벡터 컬렉션 생성

데이터 개수가 5만 건 이하로 비교적 적고, 검색 정확도가 목숨보다 중요할 때 사용하는 튜닝 코드입니다.

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

client = weaviate.connect_to_local()

try:
    client.collections.create(
        name="HighAccuracyDoc",
        # 외부 임베딩 주입 시 내부 벡터라이저는 비활성화(none). 만약 다중 벡터 비교를 원하신다면 vector_config를 사용해야 합니다.
        vectorizer_config=Configure.Vectorizer.none(),

        # [HNSW 튜닝] 정확도 극대화를 위한 도로망 촘촘화
        vector_index_config=Configure.VectorIndex.hnsw(
            max_connections=64,               # 기본값 32 -> 64 상향 (더 많은 링크)
            ef_construction=256,              # 기본값 128 -> 256 상향 (고품질 인덱싱)
            ef=128,                           # 기본값 -1 -> 128 고정 (정밀 탐색)
            distance_metric=VectorDistances.COSINE
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT)
        ]
    )
    print("✅ 고정밀 HNSW 컬렉션 생성 완료!")
finally:
    client.close()
```

---

## 3. 메모리(RAM) 비용 절감의 마법: 양자화(Quantization)와 2단계 검색

데이터 규모가 100만 건(1M) 단위를 넘어서는 순간, `float32` 타입의 고차원 벡터 인덱스는 무지막지하게 RAM을 할당해줘야 합니다. 이때 서버 비용 폭탄을 막기 위해 사용하는 것이 바로 **양자화(Quantization, 압축)** 기술입니다.

#### PQ (Product Quantization)

벡터를 잘게 쪼갠 뒤 유사한 패킷끼리 묶어 요약된 번호로 저장하여 메모리를 최대 80% 아낍니다.

#### BQ (Binary Quantization)

모든 복잡한 실수를 플러스(+)면 1, 마이너스(-)면 0으로 퉁쳐버리는 극한의 1비트 압축입니다. OpenAI의 `text-embedding-3-small` 같은 최신 모델과 궁합이 환상적입니다. 
#### 애초에 OpenAI 임베딩이 이러한 BQ를 염두하고 임베딩 모델을 만들었기에 +-에 매우 높은 가중치를 줬다고 합니다

#### SQ (Scalar Quantization)

`float32`를 `int8`로 변환하는 4배 압축. 정확도 손실이 가장 적어 BQ가 부담스러울 때의 절충안입니다.

### 양자화 기법 비교표

| 기법 | 압축률 | 정확도 손실 | 권장 시나리오 |
| :--- | :--- | :--- | :--- |
| **None (float32)** | 1x | 없음 | ~50만 건 이하, 정확도 최우선 |
| **SQ (Scalar)** | 4x | 매우 적음 | 100만~500만 건, 균형 잡힌 선택 |
| **PQ (Product)** | ~10x | 중간 | 수백만 건, 미세 튜닝이 가능할 때 |
| **BQ (Binary)** | 32x | 큼 (Rescoring으로 만회) | 1,000만 건+, 최신 임베딩 모델 사용 시 |

### ❓ 부동소수점을 0과 1로 다 깎아버리면 정확도가 박살 나지 않나요?

Weaviate는 속도와 비용은 아끼되 품질은 사수하기 위해 **2단계 검색(Over-sampling & Rescoring)** 전략을 수행합니다.

1. **1단계 (RAM 영역):** 메모리에는 압축된 BQ 벡터만 올려두고 Hamming Distance 연산으로 초고속 검색을 돌려 정답 후보군을 넉넉하게 추출합니다. (예: Top 100 오버샘플링)
2. **2단계 (디스크 영역):** 추출된 100개 후보에 대해서만 NVMe SSD 디스크에 저장되어 있던 100% 원본 고정밀 벡터를 꺼내와 코사인 유사도를 재계산(Rescoring)한 뒤 최종 Top 5를 정렬합니다.

예를 들어, RAG에 필요한 문서가 5개일 때 : 양자화된 벡터로 빠르고 가볍게 후보군을 50개 뽑기 -> 양자화 되지 않은 raw 벡터 비교로 정밀 검색하여 5개 뽑기 순서로 진행된다고 보면 됩니다.

### 💻 예시 코드 2: 대규모 데이터를 위한 이진 양자화(BQ) 컬렉션 생성

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local()

try:
    client.collections.create(
        name="MassiveDataBQ",
        vectorizer_config=Configure.Vectorizer.none(),

        # [BQ 튜닝] 메모리를 절약을 위한 이진 양자화 설정.
        vector_index_config=Configure.VectorIndex.hnsw(
            # BQ를 활성화하고 디스크 원본 벡터 재채점(rescore)을 True로 켭니다.
            quantizer=Configure.VectorIndex.Quantizer.bq(rescore_limit=200),
            # BQ로 인한 왜곡을 방지하기 위해 쿼리 탐색 범위(ef)를 넉넉하게 상향합니다.
            ef=256
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT)
        ]
    )
    print("✅ 대규모용 BQ 컬렉션 생성 완료!")
finally:
    client.close()
```

> ⚠️ **주의:** `rescore_limit`은 1단계에서 가져올 후보 개수로, `limit`보다 충분히 커야 정확도가 보장됩니다. 일반적으로 `limit`의 5~10배를 권장합니다.

---

## 4. `vectorizer_config` vs `vector_config` 문법 차이

Weaviate v4 문법으로 넘어오면서 단수형인 `vectorizer_config`와 복수형 체계인 `vector_config`가 너무 헷갈리더라구요. 알아보니 **다중 임베딩의 비교를 하느냐 마느냐**의 차이입니다.

#### `vectorizer_config` (단수형 기술 설정)

"이 테이블은 무조건 딱 하나의 벡터만 쓸 것이고 기술은 이걸로 통일하겠다!" 라는 뜻입니다. 인자에 딕셔너리를 절대 쓸 수 없으며, `Configure.Vectorizer.none()` 같은 단일 객체만 받습니다.

#### `vector_config` (복수형 공간 설정 - Named Vectors)

"이 테이블 안에는 '제목용 방', '본문용 방' 등 여러 벡터 레이어를 쪼개서 멀티로 관리하겠다!" 라는 뜻입니다. 무조건 딕셔너리 구조(`{"이름": 값}`)를 필수 수용합니다. 외부 임베딩을 직접 주입할 때는 내부의 각 설정을 `Configure.Vectors.self_provided()`로 명시해야 합니다.

### 💻 예시 코드 3: 두 방식의 문법 비교

```python
# ❌ 에러 발생 형태 (vectorizer_config 에 딕셔너리를 대입하면 TypeError가 납니다.)
"""
client.collections.create(
    name="BadCollection",
    vectorizer_config={"title_vec": Configure.Vectorizer.none()}
)
"""

# ⭕ 올바른 형태 (여러 필드의 멀티 벡터를 다룰 때는 오직 vector_config 변수만 사용!)
client.collections.create(
    name="GoodNamedVectorCollection",
    vector_config={
        "title_vector": Configure.Vectors.self_provided(),   # 제목용 외부 임베딩 방
        "content_vector": Configure.Vectors.self_provided()  # 본문용 외부 임베딩 방
    },
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT)
    ]
)
```

---

## 5. 다중 타겟 벡터 검색(Multi-Target Search)과 품질 저하 방어선

여러 개의 벡터 필드(`vector_config`)를 구축해 두면, 유저가 검색창에 입력한 단 하나의 질문 벡터를 가지고 **제목 인덱스와 본문 인덱스를 실시간으로 동시에 타격**할 수 있습니다.

하지만 아무런 전략 없이 두 공간에서 나온 스코어를 1:1 단순 평균으로 합산하면 **정보의 희석(Noise Amplification)** 현상이 일어납니다. 본문 구석에 기가 막힌 정답이 있어서 본문 벡터 점수는 만점인데, 제목은 엉뚱한 소리라 점수가 빵점이라면, 평균을 내는 순간 정답 문서가 순위권 뒤로 밀려버립니다.

Weaviate는 이 결합 리스크를 방어하기 위한 정교한 **Join Strategy** 함수를 제공합니다.

#### `TargetVectors.minimum()` (강력 추천)

거리 점수 중 가장 작은 값(유사도가 가장 높은 값) 하나만 최종 점수로 채택합니다. 즉, 제목이든 본문이든 어느 한 곳에서만 스나이핑에 성공하면 점수 삭감 없이 합격시키는 강력한 방어선입니다.

#### `TargetVectors.manual_weights()`

개발자가 직접 제목 매칭에 70%(0.7), 본문 매칭에 30%(0.3)의 가중치를 배정하여 본문의 노이즈 단어들이 전체 점수를 갉아먹지 못하도록 힘을 실어주는 방식입니다.

#### `TargetVectors.relative_score()`

각 벡터 공간 내에서 상대적 순위를 기반으로 점수를 정규화한 뒤 결합합니다. 임베딩 모델이 서로 다른 분포를 가진 멀티 벡터 환경에 효과적입니다.

#### `TargetVectors.sum()` / `average()`

단순 합산/평균. 모든 벡터 공간의 신뢰도가 비슷할 때만 안전합니다.

### 💻 예시 코드 4: 결합 전략을 조작한 다중 타겟 쿼리

```python
from weaviate.classes.query import TargetVectors

collection = client.collections.get("GoodNamedVectorCollection")

# 유저 검색어로 뽑아낸 단 하나의 1536차원 쿼리 벡터 (One-Source, Multi-Use)
user_query_vector = [0.015, -0.022, 0.114] + [0.0] * 1533

response = collection.query.near_vector(
    near_vector=user_query_vector,

    # [품질 방어] 본문의 긴 내용 때문에 점수가 희석되는 것을 minimum() 전략으로 완벽 방어
    target_vector=TargetVectors.minimum([
        "title_vector",
        "content_vector"
    ]),
    limit=5
)

for obj in response.objects:
    print(f"🎯 최종 노출 문서 제목: {obj.properties['title']}")
```

---

## 6. 아키텍처 제언: 제목 임베딩 vs LLM 키워드 추출

많은 RAG 개발자들이 *"본문이 전부인데 굳이 제목까지 따로 임베딩을 써야 하나? 차라리 LLM을 거쳐 핵심 키워드 메타데이터를 뽑아서 키워드 필터링을 거는 게 낫지 않나?"* 라는 아키텍처 고민을 합니다. 네.. 저예요....

실무 프로덕션 관점에서는 '비용 가성비'와 '형태적 유사성' 때문에 **제목 임베딩을 베이스로 깔고 가는 것이 무조건 이득**입니다.

#### 비용의 격차

1,500토큰 분량의 문서 5만 건을 LLM(예: GPT-4o)에 밀어 넣어 "핵심 키워드 뽑아줘"라고 처리하려면 무려 7,500만 토큰의 비용이 소모되고 레이턴시가 폭증합니다. 반면 기존의 제목 텍스트만 쏙 뽑아 임베딩하는 비용은 얼마 안되죠.

#### 검색 형태의 싱크로율

유저가 검색창에 던지는 쿼리 문장(예: "~하는 방법 가이드")은 장황한 설명문 형태의 본문보다 요약형 문장 구조인 '제목'과 형태적으로 가장 닮아 있습니다. 벡터 공간에서는 어조와 구조가 닮을수록 점수가 높게 찍히므로 매칭률이 극대화됩니다.

### 💡 하이브리드 아키텍처

가장 이상적인 타협점은 두 기술의 역할을 아래와 같이 **완전히 분리**하는 것입니다.

1. **벡터 검색 영역:** 이미 존재하는 `[제목]`과 `[본문]`을 각각 임베딩하여 다중 타겟 벡터 인덱스로 구축합니다.
2. **LLM 태깅 영역:** LLM에게는 새로운 텍스트 생성을 시키지 말고, 텍스트 일치 필터링용 `[정형 태그(카테고리, 중요도, 부서 등)]`만 가볍게 추출하게 합니다.
3. **융합 검색 영역:** Weaviate의 `hybrid()` 검색을 활용해 `[다중 벡터 유사도]` + `[LLM이 뽑아준 정형 태그/텍스트 검색(BM25)]`을 5:5 비율로 결합합니다.

---

## 7. 프로덕션 엔지니어를 위한 숨겨진 고도화 옵션 3종 세트

여기서 한 걸음 더 나아가, 대형 포털이나 커머스 인프라 급의 성능을 내기 위해 Weaviate 컬렉션 생성 시 반드시 세팅해야 할 **숨겨진 끝판왕 설정 3가지**를 공유합니다. 저도 이것까진 안해봐서 AI로만 정리해봤어요.

### ① 역색인 다이어트 (`index_filterable`, `index_searchable`)

Weaviate는 필드를 만들면 **조건 필터용(`where`) 인덱스**와 **키워드 검색용(BM25) 인덱스**를 모두 자동 빌드합니다. 만약 본문(content) 필드로 `=` 조건 비교 필터링을 걸 일이 전혀 없고 오직 하이브리드 문자열 검색과 최종 출력용으로만 쓴다면, 필터 인덱스를 수동으로 꺼서 디스크 용량을 대폭 절약하고 삽입 속도를 올릴 수 있습니다.

### ② OOM 방어선 구축 (`vector_cache_max_objects`)

클라우드 가성비 서버를 쓸 때 데이터가 늘어나면 서버가 갑자기 OOM으로 크래시가 납니다. HNSW 설정 내부에 이 옵션을 주면, *"RAM 메모리에는 딱 이 개수만큼의 벡터만 올려두고 넘치는 부분은 디스크(NVMe)에서 실시간 스왑하며 읽어와라"* 라고 강제 제동을 걸 수 있어 서버 다운을 원천 차단합니다.

### ③ 리랭커(Reranker) 모듈 연동

하이브리드(벡터+BM25) 스코어가 완벽하진 않기에, 1차로 Weaviate가 상위 50개 문서를 초고속으로 추려내게 만든 뒤, 2차로 **문맥 파악 전문 모델인 교차 인코더(Cross-Encoder) 기반의 Reranker 모듈**을 통과시켜 최상의 고품질 Top 5로 정렬하는 파이프라인 기법입니다.

> ⚠️ **주의:** `Configure.Reranker.cohere()` 등의 리랭커 모듈은 Weaviate 서버 기동 시 `ENABLE_MODULES` 환경변수(예: `reranker-cohere`)에 등록되어 있어야 하며, 해당 API 키도 컨테이너에 주입되어야 정상 작동합니다.

---

## 8. 팩토리 레벨: 완벽하게 최적화된 마스터 프로덕션 템플릿

위에서 언급한 HNSW 조작, 멀티 타겟 방 쪼개기, 역색인 다이어트, OOM 방어선, 그리고 리랭커 모듈 연동까지 **단 하나의 파이썬 스크립트로 결합한 최종 마스터 코드**입니다. 실무 프로젝트 구축 시 이 템플릿을 베이스로 삼으시면 됩니다.

### 💻 마스터 코드: 스키마 생성, 초고속 대량 삽입(Batch), 고도화 검색 쿼리

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import TargetVectors, Rerank

# 1. Weaviate 인스턴스 연결
client = weaviate.connect_to_local()

try:
    # 2. [종합 고도화 설계] 컬렉션 생성
    if client.collections.exists("ProductionMasterDoc"):
        client.collections.delete("ProductionMasterDoc")

    client.collections.create(
        name="ProductionMasterDoc",
        description="모든 최적화 기법이 융합된 팩토리 레벨 컬렉션",

        # [고도화 1] Cohere 리랭커 모듈 원격 탑재 설정
        reranker_config=Configure.Reranker.cohere(),

        # [HNSW + 멀티벡터 + OOM 방어] 통합 제어
        vector_config={
            "title_vector": Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    max_connections=64,
                    ef_construction=256,
                    vector_cache_max_objects=500000  # [고도화 2] RAM 캐싱 한계 설정 (OOM 방지)
                )
            ),
            "content_vector": Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    max_connections=64,
                    ef_construction=256,
                    vector_cache_max_objects=500000  # RAM 캐싱 마지노선 구축
                )
            )
        },

        # [고도화 3] 역색인 구조 최적화 (인덱스 다이어트)
        properties=[
            Property(
                name="title",
                data_type=DataType.TEXT,
                index_filterable=True,   # 제목은 필터링 조건문(=)에 자주 쓰이므로 True
                index_searchable=True
            ),
            Property(
                name="content",
                data_type=DataType.TEXT,
                index_filterable=False,  # 본문은 덩치가 커서 필터링 인덱스 구축을 제외시켜 디스크 절약!
                index_searchable=True    # 하이브리드 키워드 검색에는 쓰여야 하므로 True
            ),
            Property(
                name="category",
                data_type=DataType.TEXT, # LLM 이 태깅한 정형 필터링용 메타데이터 방
                index_filterable=True,
                index_searchable=False   # 검색용이 아니라 조건 매칭용이므로 검색 인덱스 해제
            )
        ]
    )
    print("🚀 마스터 최적화 컬렉션 빌드 완료!")

    # 3. [초고속 대량 삽입] 외부 벡터 매핑 및 Fixed-Size 배치 적재
    collection = client.collections.get("ProductionMasterDoc")

    # 5만 건 적재 파이프라인 시뮬레이션용 모크 데이터 (외부 임베딩 완료 상태 가정)
    sample_batch_data = [
        {
            "title": f"대규모 시스템 튜닝 가이드 {i}",
            "content": f"이것은 약 1500토큰 분량의 긴 문서 본문입니다. Weaviate 최적화 팁 번호 {i}",
            "category": "infra",
            "title_vec": [0.01] * 1536,
            "content_vec": [-0.02] * 1536
        } for i in range(200)  # 200개 단위 예시 샘플
    ]

    # 네트워크 오버헤드를 줄이기 위한 배치 컨텍스트 오픈
    with collection.batch.fixed_size(batch_size=100) as batch:
        for doc in sample_batch_data:
            batch.add_object(
                properties={
                    "title": doc["title"],
                    "content": doc["content"],
                    "category": doc["category"]
                },
                # 멀티 벡터 환경이므로 각 방의 이름을 키값으로 지정하여 벡터 주입
                vector={
                    "title_vector": doc["title_vec"],
                    "content_vector": doc["content_vec"]
                }
            )

    # 배치 적재 후 실패 객체 확인 (운영 단계 필수 점검)
    failed_objects = collection.batch.failed_objects
    if failed_objects:
        print(f"⚠️ 배치 실패 객체 {len(failed_objects)}건 발견 — 재시도 필요")
    else:
        print(f"📦 {len(sample_batch_data)}건 데이터 고속 배치 삽입 성공!")

    # 4. [종합 고도화 쿼리] 하이브리드 + 멀티타겟 Minimum 결합 + 리랭커 재정렬
    user_query_vec = [0.01] * 1536  # 단 하나의 유저 질문 쿼리 벡터

    response = collection.query.hybrid(
        query="대규모 시스템 최적화 튜닝",  # 키워드(BM25)용 검색 문자열
        vector=user_query_vec,             # 의미 검색용 쿼리 벡터
        alpha=0.5,                         # 벡터와 키워드 가중치 반반 융합

        # 멀티 타겟 스코어 결합 전략 제어 (희석 현상 방어)
        target_vector=TargetVectors.minimum([
            "title_vector",
            "content_vector"
        ]),

        # [품질의 끝판왕] 상위 결과를 Cohere 딥러닝 문맥 기반 모델로 2차 최종 재정렬
        rerank=Rerank(
            prop="content",                # 본문 문맥을 기준으로 리랭킹 수행
            query="대규모 시스템 최적화 튜닝"
        ),
        limit=5
    )

    # 5. 최종 결과 출력
    print("\n🔍 초고품질 검색 결과 리스트:")
    print("=" * 50)
    for obj in response.objects:
        print(f"📄 제목: {obj.properties['title']}")
        print(f"🏷️ 카테고리: {obj.properties['category']}")
        print(f"📊 최종 융합 스코어: {obj.metadata.score}")
        print("-" * 50)

finally:
    client.close()
```

---

## 9. 보너스: 비동기(Async) 클라이언트로 처리량 끌어올리기

RAG 서비스가 FastAPI 기반의 API 서버라면, 동기(sync) 클라이언트보다 **비동기(async) 클라이언트**가 동시 요청 처리량에서 압도적으로 유리합니다. 특히 임베딩 API 호출과 Weaviate 검색을 `asyncio.gather()`로 묶으면 I/O 대기 시간을 거의 0으로 만들 수 있습니다. 저희 회사에서는 FastAPI 기반의 서버를 운영하기에 이런 방식으로 사용하고 있어요.

```python
import asyncio
import weaviate

async def search_async(query_vec: list[float]):
    async with weaviate.use_async_with_local() as client:
        collection = client.collections.get("ProductionMasterDoc")
        response = await collection.query.near_vector(
            near_vector=query_vec,
            limit=5
        )
        return response.objects

# 여러 쿼리를 동시 발사
async def main():
    queries = [[0.01] * 1536 for _ in range(10)]
    results = await asyncio.gather(*[search_async(q) for q in queries])
    return results

# asyncio.run(main())
```

---

## 마치며

그냥 단순하게 하이브리드 서치를 지원하는구나! 했던 Weaviate가 파면팔수록 뭐가 많이 나오더라구요. 해서 실제 회사에서 돌아가고 있는 서비스도 열어봤는데 이정도 고도화는 되어 있는것도 있고 아닌것도 있기에 적재량이나 서비스 이용량에 따라 적절히 선택하는게 핵심일 것 같습니다.