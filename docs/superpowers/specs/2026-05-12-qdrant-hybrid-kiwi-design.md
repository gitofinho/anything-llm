# Qdrant Hybrid Search with Kiwi (Korean Morphological Tokenizer) — Design

**Date**: 2026-05-12
**Status**: Approved — pending implementation plan
**Scope**: AnythingLLM 본체 정식 통합 (fork patch level, PR 가능)

## Motivation

AnythingLLM의 Qdrant 벡터 DB 프로바이더는 현재 단일 dense 임베딩 + Cosine
유사도만 사용한다. 한국어 도메인(예: 한국주택금융공사 채용 공고처럼 고유명사·
복합명사·조사 비중이 높은 문서)에서는 dense-only 검색이 정확한 키워드
매칭에 약하다. 이 작업은:

1. Qdrant의 named-vector + sparse + Query API 기반 hybrid 검색(RRF fusion)을 도입.
2. 한국어 형태소 분석기 Kiwi를 사이드카로 운영해 인덱싱·질의 양쪽에서 동일한
   토큰화를 적용.
3. 위 두 가지를 옵션 플래그로 기존 사용자 데이터에 무영향으로 점진 도입.

목적은 챗 위젯(`embed/`)을 통한 한국어 도메인 응답 품질의 retrieval 단계 개선.
위젯 UI는 변경하지 않는다.

## Non-goals (YAGNI)

- 다국어 토크나이저 추상화 (한국어 외 텍스트는 dense 위주로 자연스럽게 동작)
- Kiwi 사용자 사전 커스터마이즈
- 학습된 sparse (SPLADE 등)
- 워크스페이스 단위 IDF (collection 단위만)
- 챗 위젯 UI 변경
- 기존 dense-only collection의 자동 마이그레이션
- Qdrant 이외의 vector DB 프로바이더 변경

## Architecture

```
┌──────────────┐     ┌──────────────────────────────┐
│  embed 위젯   │ ─▶ │  server (Node)                │
└──────────────┘     │  vectorDbProviders/qdrant     │
                     │  ├ addDocumentToNamespace ────┼──┐ POST /tokenize
                     │  └ performSimilaritySearch ───┼──┤ (batch)
                     └──────────────────────────────┘  │
                                  ▲                     ▼
                     ┌──────────────────┐    ┌──────────────────────┐
                     │ collector (Node)  │   │ kiwi-service          │
                     │  parses files,    │   │ FastAPI + kiwipiepy   │
                     │  emits chunks     │   │ /tokenize  /healthz   │
                     └──────────────────┘    └──────────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────────────┐
                     │  Qdrant (named vectors:         │
                     │  {dense, sparse}) + RRF fusion  │
                     └────────────────────────────────┘
```

Collector는 기존대로 파일 파싱·청크 메타만 산출하고, Kiwi 토큰화·sparse 빌드·
upsert·검색은 모두 server 의 Qdrant provider 안에서 일어난다.

### Components — new

- **`kiwi-service/`** (repo root): 독립 Python 사이드카.
  - `app.py` — FastAPI. `POST /tokenize` 배치 호출, `GET /healthz`.
  - `Dockerfile` — `python:3.12-slim` 기반 + `kiwipiepy`.
  - `requirements.txt`, `tests/` (pytest).
  - 기본 포트 `8765`.
- **`docker-compose.yml`** 에 `kiwi-service` 항목 추가 (compose profile `hybrid`로 옵셔널).

### Components — changed

- **`server/utils/vectorDbProviders/qdrant/index.js`** — 모든 Kiwi 호출과
  hybrid 분기가 이 한 파일에 응집된다.
  - `getOrCreateCollection`: hybrid 플래그 + kiwi healthy 조건 충족 시 named
    vectors(`dense`, `sparse`) 로 컬렉션 생성.
  - `namespace`/`hasNamespace` 등에서 컬렉션 vector 스키마(`config.params.vectors`
    모양)를 캐시. 기존 unnamed dense 컬렉션은 자동 dense-only 경로.
  - `addDocumentToNamespace`: 청크 생성·임베딩 직후 Kiwi 토큰화 → 품사 필터
    → doc-side BM25 TF → sparse `(indices, values)` 빌드 → `__bm25_stats__`
    포인트 갱신 → `client.upsert` payload 에 sparse 포함.
  - `performSimilaritySearch` / 내부 `client.search` 호출: hybrid 컬렉션이면
    질의 토큰화 후 `client.query` + `prefetch[dense, sparse]` + `fusion=rrf`.
- **`server/utils/helpers/index.js`** (또는 동등 환경 설정 모듈): 새 환경
  변수 export 및 startup 로그(`hybrid: enabled/disabled, kiwi: healthy/...`).
- **신규 헬퍼**: `server/utils/vectorDbProviders/qdrant/hybrid/` 하위에
  `kiwiClient.js` (HTTP + health 캐시), `bm25.js` (TF/IDF·sparse builder),
  `stats.js` (`__bm25_stats__` 포인트 read/upsert + 직렬화 mutex).
- **collector 는 변경 없음**.

### Environment variables (new)

| Name | Default | 의미 |
|---|---|---|
| `QDRANT_HYBRID_ENABLED` | `false` | 신규 collection을 hybrid 스키마로 생성할지 |
| `KIWI_SERVICE_URL` | `http://kiwi-service:8765` | kiwi 사이드카 base URL |
| `QDRANT_HYBRID_FUSION` | `rrf` | `rrf` 만 지원 (MVP) |
| `QDRANT_HYBRID_BM25_K1` | `1.2` | BM25 k1 |
| `QDRANT_HYBRID_BM25_B`  | `0.75` | BM25 b |

## Data flow

### Indexing (server `addDocumentToNamespace` → Qdrant)

Collector 는 기존대로 파일을 파싱해 청크 메타를 내려보내고, 아래는 server
측 `addDocumentToNamespace` 내부 흐름.

1. 청크 텍스트 생성 (기존 `TextSplitter` 그대로).
2. dense 임베딩 생성 (기존 embedder).
3. `POST kiwi-service/tokenize` 배치: `{ texts: [...], filterPos: ["NNG","NNP","SL","SH","SN"] }` →
   `{ tokens: [["주택","금융","공사",...], ...] }`.
4. 청크별 token frequency 계산. token → `hash(token) mod 2^20` (FNV-1a 32bit)
   으로 sparse index 생성 (충돌 무시; BM25 근사 sparse에서 흔한 패턴).
5. 컬렉션의 `__bm25_stats__` 예약 포인트 갱신 — `df` 맵(token → 문서 수),
   `N`(총 문서 수), `avgdl`(평균 청크 길이). 동시성은 컬렉션 단위 in-process
   mutex 로 직렬화 (멀티 인스턴스 케이스는 Risks 항 참고).
6. `client.upsert(namespace, { points: [{ id, vector: { dense, sparse: { indices, values } }, payload }] })`.

### Query (server → Qdrant)

1. 사용자 질의 → kiwi-service 토큰화 (인덱싱과 동일 품사 필터).
2. dense 임베딩 생성 (기존).
3. Query-side BM25 weight = `idf(t)` (질의 토큰은 보통 tf=1):
   `idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)`.
4. `client.query(namespace, { prefetch: [
     { using: 'dense',  query: [...], limit: 50 },
     { using: 'sparse', query: { indices, values }, limit: 50 }
   ], query: { fusion: 'rrf' }, limit: topK })`.
5. 결과 → 기존 LLM 컨텍스트 조립.

### Backward compatibility

`getOrCreateCollection` 진입 시:

- **신규** collection + `QDRANT_HYBRID_ENABLED=true` + kiwi healthy
  → named vectors `{dense, sparse}`.
- **신규** + flag off (or kiwi unhealthy) → 기존 unnamed dense (legacy 경로).
- **기존** unnamed-dense collection → 형식 유지. 검색 시 자동 dense-only
  분기, "hybrid disabled for legacy collection X" 경고 1회 로그.

컬렉션 스키마 판정은 `client.getCollection()` 결과의 `config.params.vectors`
모양 (object vs unnamed) 으로 캐시.

## Error handling

- **kiwi-service unreachable**:
  - 인덱싱 시: 재시도 3회(지수 backoff 0.5/1/2s) 후 실패 전파. 부분 일관성
    회피.
  - 검색 시: dense-only 로 fallback + 경고 로그. health 체크는 5초 캐시로
    throttle.
- **BM25 stats point 부재**: 새 컬렉션이므로 빈 IDF로 시작, 인덱싱하며 누적.
  검색 시 IDF=0 인 토큰은 sparse leg 점수 0 (사실상 무시).
- **한국어 외 질의** (Kiwi 토큰 결과가 empty): sparse leg skip, dense-only
  fusion path (점수상 dense 결과만).
- **Qdrant 1.10+ 미만 환경**: Query API 미지원. Provider 초기화 시 cluster
  버전 체크하여 hybrid 강제 off + 경고. README에 최소 버전 명시.

## Sample data

```
samples/qdrant-hybrid-kiwi/
├── README.md
├── raw/
│   └── 2026_hf_recruitment.pdf  # gitignored (외부 출처, 바이너리)
└── fixtures/
    └── expected_queries.json    # 통합 테스트용 (질의 + 기대 키워드)
```

`raw/` 는 `.gitignore` 에 등록 (`samples/**/raw/`). README에 로컬 복사 방법
명시. fixtures 는 commit.

## Testing strategy

- **Unit**
  - `kiwi-service`: `/tokenize` happy path + 빈 입력 + 비-한국어 + 품사 필터.
  - BM25 sparse builder: 토큰 → sparse `(indices, values)`, idf 계산식.
  - Qdrant provider: collection 스키마 분기 (unnamed vs named) snapshot test.
- **Integration** (compose 기반)
  - `qdrant` + `kiwi-service` 컨테이너 + collector + server 로컬 기동.
  - 샘플 PDF ingest → fixture 5개 질의 → top-5 안 기대 키워드 hit-rate
    측정. **Acceptance**: hybrid hit-rate ≥ dense-only baseline (동일 환경,
    같은 임베딩 모델). 회귀 방지 목적이라 절대치 목표는 두지 않음.
- **Regression**
  - 기존 Qdrant unnamed-vector 경로 unit test 100% pass (legacy 컬렉션 시
    분기가 정확히 dense-only 인지).
- **Manual smoke**
  - 위젯 로컬 기동 + ingest 한 워크스페이스 한국어 질의. UI 무변화 확인.

## Git strategy

```
master
  └─ feature/qdrant-hybrid-kiwi          (long-lived 통합 브랜치)
       ├─ feat/kiwi-service              (PR #1 → 통합)
       ├─ feat/qdrant-hybrid-schema      (PR #2 → 통합)
       ├─ feat/qdrant-hybrid-index       (PR #3 → 통합, #2 의존; server addDocumentToNamespace)
       ├─ feat/qdrant-hybrid-query       (PR #4 → 통합, #2 의존)
       └─ feat/qdrant-hybrid-sample-e2e  (PR #5 → 통합, 모두 의존)
```

- 각 sub-PR 100–400 LOC 목표, 독립 리뷰.
- 통합 브랜치 e2e 그린 후 squash-merge 로 master (또는 upstream fork main).
- 커밋 컨벤션: 기존 repo 스타일 (`feat: …`, `feat(qdrant): …`, `fix: …`).
- 통합 브랜치는 주 1회 `git merge master` 로 sync. rebase 금지 (sub-PR
  origin 보존).

## Sub-agent decomposition

5개 역할. 의존 그래프대로 3개 라운드(2 → 2 → 1) 로 병렬 디스패치.

| Agent | 책임 | 산출물 | Deps |
|---|---|---|---|
| `kiwi-service-agent` | Python 사이드카 | `kiwi-service/` (FastAPI, Dockerfile, pytest), compose 항목 | — |
| `qdrant-schema-agent` | Qdrant provider 스키마 계층 | `qdrant/index.js` 의 collection 생성/스키마 감지/캐시 + hybrid helper 디렉터리(`hybrid/{kiwiClient,bm25,stats}.js`) 스캐폴드, unit test | — |
| `server-indexing-agent` | server 인덱싱 파이프라인 | `addDocumentToNamespace` 안에서 토큰화 호출 + sparse builder + `__bm25_stats__` 갱신 + upsert payload 변경 | schema, kiwi |
| `server-query-agent` | server 질의 파이프라인 | 검색 시 토큰화 + Qdrant Query API + RRF fusion + dense fallback | schema, kiwi |
| `integration-test-agent` | E2E | compose 통합 테스트, hit-rate 비교 리포트 | 전부 |

- **Round 1 (병렬)**: `kiwi-service-agent` ‖ `qdrant-schema-agent`
- **Round 2 (병렬)**: `server-indexing-agent` ‖ `server-query-agent`
- **Round 3 (단독)**: `integration-test-agent`

각 sub-agent 는 `superpowers:test-driven-development` 따름.
실행 단계에서 `superpowers:dispatching-parallel-agents` 가 라운드 내
병렬화를 담당.

## Acceptance criteria

1. `QDRANT_HYBRID_ENABLED=false` 환경에서 기존 Qdrant 사용자 경로(unnamed dense
   collection, dense-only 검색)가 100% 회귀 없음.
2. `QDRANT_HYBRID_ENABLED=true` + kiwi healthy + 새 워크스페이스 생성 시
   collection 이 named vectors 로 만들어지고, ingest/query 가 hybrid 경로로
   동작.
3. 기존 dense-only collection을 가진 워크스페이스가 hybrid 활성 후에도 정상
   동작 (자동 fallback + 경고 로그).
4. 샘플 PDF 통합 테스트에서 hybrid hit-rate ≥ dense-only baseline.
5. kiwi-service 다운 시 검색 경로 가 dense-only 로 graceful degrade.

## Risks & open questions

- **Qdrant 버전 최소요건**: Query API + sparse named vectors 는 1.10+. 사용
  중인 client SDK (`@qdrant/js-client-rest`) 의 호환성 확인 필요 — 구현
  플랜에서 actual 버전 핀과 호환성 검증을 첫 task로.
- **IDF 통계 동시성**: 멀티 워커가 동시에 같은 컬렉션에 ingest 할 때 `__bm25_stats__`
  포인트 갱신이 race 함. MVP는 server-process 단일 mutex; 멀티 인스턴스 배포
  케이스는 별도 후속 작업.
- **Compose profile UX**: 사용자가 `--profile hybrid` 없이 띄우면 server는
  kiwi-service에 도달 못해 검색이 dense-only 로 떨어짐. 동작은 정상이지만
  사용자에게 명확한 시그널(시작 로그)이 필요.
- **PDF 라이선스**: 한국주택금융공사 공개 게재물이나, 재배포 가능 여부 미확정 →
  `samples/**/raw/` gitignore 로 안전 측.
