# Qdrant Hybrid Search + Kiwi Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AnythingLLM의 Qdrant 프로바이더에 한국어 형태소(Kiwi) 기반 BM25 sparse + dense hybrid 검색(Query API · RRF fusion)을 옵션 플래그로 도입하고, 챗 위젯의 한국어 retrieval 품질을 회귀 없이 개선한다.

**Architecture:** Python `kiwipiepy` 사이드카(`kiwi-service`)가 토큰화 책임을 단일화하고, server 의 Qdrant 프로바이더(`addDocumentToNamespace` · `similarityResponse`) 가 인덱싱·질의 양쪽에서 이를 호출한다. 컬렉션은 named vectors `{dense, sparse}` 로 생성하되, 기존 unnamed-dense 컬렉션은 자동 dense-only 분기로 fallback.

**Tech Stack:** Node.js 18+ (server), `@qdrant/js-client-rest` (≥1.10), Python 3.12 + FastAPI + `kiwipiepy`, Qdrant ≥1.10, Jest (server unit), pytest (kiwi-service), Docker Compose.

**Reference spec:** `docs/superpowers/specs/2026-05-12-qdrant-hybrid-kiwi-design.md`

---

## File Structure

**New**
- `kiwi-service/app.py`, `kiwi-service/Dockerfile`, `kiwi-service/requirements.txt`, `kiwi-service/.dockerignore`, `kiwi-service/tests/test_app.py`, `kiwi-service/pytest.ini`
- `server/utils/vectorDbProviders/qdrant/hybrid/kiwiClient.js` — kiwi-service HTTP client + health 캐시
- `server/utils/vectorDbProviders/qdrant/hybrid/bm25.js` — sparse builder (hash, TF, IDF, query weight)
- `server/utils/vectorDbProviders/qdrant/hybrid/stats.js` — `__bm25_stats__` 포인트 read/upsert + 컬렉션 단위 mutex
- `server/utils/vectorDbProviders/qdrant/hybrid/config.js` — env 파싱(`QDRANT_HYBRID_ENABLED` 등)
- `server/__tests__/vectorDbProviders/qdrant/bm25.test.js`
- `server/__tests__/vectorDbProviders/qdrant/kiwiClient.test.js`
- `server/__tests__/vectorDbProviders/qdrant/stats.test.js`
- `server/__tests__/vectorDbProviders/qdrant/schemaBranching.test.js`
- `server/__tests__/vectorDbProviders/qdrant/queryFusion.test.js`
- `server/__tests__/integration/qdrantHybridE2E.test.js`
- `scripts/integration/start-hybrid-stack.sh` — docker-compose `--profile hybrid` 기동/정리 헬퍼

**Modified**
- `docker/docker-compose.yml` — `kiwi-service` 추가 (profile `hybrid`)
- `server/package.json` — `jest` devDep + `test` script + 클라이언트 버전 bump
- `server/utils/vectorDbProviders/qdrant/index.js` — collection 스키마 분기, hybrid 헬퍼 호출
- `server/utils/helpers/updateENV.js` — 새 env 항목 등록
- `server/.env.example` — 새 env 문서화
- `docker/.env.example` — `KIWI_SERVICE_URL` 도커 기본값 안내

---

## Branching

```
master
  └─ feature/qdrant-hybrid-kiwi          (통합 브랜치)
       ├─ feat/kiwi-service              (Task 1 → PR #1)
       ├─ feat/qdrant-hybrid-helpers     (Task 2 → PR #2)
       ├─ feat/qdrant-hybrid-schema      (Task 3 → PR #3)
       ├─ feat/qdrant-hybrid-index       (Task 4 → PR #4)
       ├─ feat/qdrant-hybrid-query       (Task 5 → PR #5)
       └─ feat/qdrant-hybrid-e2e         (Task 6 → PR #6)
```

각 Task 의 마지막 Step 은 sub-branch 에서 통합 브랜치로 PR 을 띄우는 것까지 포함한다.

---

## Task 0: Bootstrap branches and shared scaffolding

**Files:**
- Create: `feature/qdrant-hybrid-kiwi` branch
- Modify: `server/package.json`
- Modify: `server/utils/helpers/updateENV.js`
- Modify: `server/.env.example`
- Modify: `docker/.env.example`

- [ ] **Step 0.1: Create the long-lived integration branch off master**

```bash
git fetch origin
git checkout master
git pull --ff-only
git checkout -b feature/qdrant-hybrid-kiwi
git push -u origin feature/qdrant-hybrid-kiwi
```

- [ ] **Step 0.2: Create the env-bootstrap sub-branch**

```bash
git checkout -b feat/qdrant-hybrid-env-bootstrap
```

- [ ] **Step 0.3: Add jest as server devDep and a `test` script**

Edit `server/package.json`:

```json
"scripts": {
  "dev": "cross-env NODE_ENV=development nodemon --ignore documents --ignore vector-cache --ignore storage --ignore swagger --trace-warnings index.js",
  "start": "cross-env NODE_ENV=production node index.js",
  "lint": "eslint --fix .",
  "lint:check": "eslint .",
  "test": "jest --testPathIgnorePatterns=__tests__/integration",
  "test:integration": "jest --runInBand __tests__/integration",
  "swagger": "node ./swagger/init.js"
}
```

Add to `devDependencies` (alphabetized): `"jest": "^29.7.0"`.

Bump in `dependencies`: `"@qdrant/js-client-rest": "^1.10.0"` (Qdrant Query API + sparse named vectors 지원).

- [ ] **Step 0.4: Install**

```bash
cd server && yarn install
```

Expected: `Done in ...`. New `node_modules/.bin/jest` exists.

```bash
ls node_modules/.bin/jest && node_modules/.bin/jest --version
```

Expected: prints `29.x.x`.

- [ ] **Step 0.5: Add new env entries to `updateENV.js`**

In `server/utils/helpers/updateENV.js`, find the existing `QDRANT_API_KEY` block (around line 386). Add **immediately after** it:

```js
  QdrantHybridEnabled: {
    envKey: "QDRANT_HYBRID_ENABLED",
    checks: [isValidBoolStringOrNull],
  },
  KiwiServiceURL: {
    envKey: "KIWI_SERVICE_URL",
    checks: [isValidURL],
  },
  QdrantHybridFusion: {
    envKey: "QDRANT_HYBRID_FUSION",
    checks: [(input) => (["rrf", null, undefined, ""].includes(input) ? null : "Only 'rrf' is supported in this release.")],
  },
  QdrantHybridBM25K1: {
    envKey: "QDRANT_HYBRID_BM25_K1",
    checks: [isNumericStringOrNull],
  },
  QdrantHybridBM25B: {
    envKey: "QDRANT_HYBRID_BM25_B",
    checks: [isNumericStringOrNull],
  },
```

If `isValidBoolStringOrNull` / `isNumericStringOrNull` helpers don't already exist in this file, add them right above the `KEY_MAPPING` object (or wherever sibling validators live):

```js
function isValidBoolStringOrNull(input) {
  if (input === null || input === undefined || input === "") return null;
  return ["true", "false"].includes(String(input).toLowerCase()) ? null : "Must be 'true' or 'false'.";
}

function isNumericStringOrNull(input) {
  if (input === null || input === undefined || input === "") return null;
  return Number.isFinite(Number(input)) ? null : "Must be a numeric value.";
}
```

Re-use `isValidURL` if it exists; if not, add:

```js
function isValidURL(input) {
  if (!input) return null;
  try { new URL(input); return null; } catch { return "Must be a valid URL."; }
}
```

- [ ] **Step 0.6: Document new env vars**

Append to `server/.env.example`:

```
# ---- Qdrant hybrid (Korean) ----
# QDRANT_HYBRID_ENABLED=false
# KIWI_SERVICE_URL=http://localhost:8765
# QDRANT_HYBRID_FUSION=rrf
# QDRANT_HYBRID_BM25_K1=1.2
# QDRANT_HYBRID_BM25_B=0.75
```

Append to `docker/.env.example`:

```
# Set to http://kiwi-service:8765 when running with `--profile hybrid`.
# KIWI_SERVICE_URL=
# QDRANT_HYBRID_ENABLED=false
```

- [ ] **Step 0.7: Smoke-test jest discovery**

```bash
cd server && yarn test --listTests | head -5
```

Expected: prints paths to existing `__tests__/**/*.test.js` files. No errors.

- [ ] **Step 0.8: Commit and PR**

```bash
git add server/package.json server/yarn.lock server/utils/helpers/updateENV.js server/.env.example docker/.env.example
git commit -m "feat(qdrant): bootstrap hybrid env vars and dev test runner"
git push -u origin feat/qdrant-hybrid-env-bootstrap
gh pr create --base feature/qdrant-hybrid-kiwi --title "feat(qdrant): bootstrap hybrid env vars and dev test runner" --body "Adds new env entries (QDRANT_HYBRID_ENABLED, KIWI_SERVICE_URL, fusion/BM25 params), Jest dev runner, and Qdrant client bump to ^1.10.0 in prep for hybrid search. No runtime behavior change."
```

After merge of PR back into `feature/qdrant-hybrid-kiwi`, return to integration branch:

```bash
git checkout feature/qdrant-hybrid-kiwi && git pull --ff-only
```

---

## Task 1: `kiwi-service` Python sidecar

**Files:**
- Create: `kiwi-service/app.py`
- Create: `kiwi-service/Dockerfile`
- Create: `kiwi-service/requirements.txt`
- Create: `kiwi-service/.dockerignore`
- Create: `kiwi-service/pytest.ini`
- Create: `kiwi-service/tests/__init__.py`
- Create: `kiwi-service/tests/test_app.py`
- Modify: `docker/docker-compose.yml`

- [ ] **Step 1.1: Branch**

```bash
git checkout feature/qdrant-hybrid-kiwi
git checkout -b feat/kiwi-service
```

- [ ] **Step 1.2: Write the failing API tests**

`kiwi-service/tests/test_app.py`:

```python
from fastapi.testclient import TestClient
from kiwi_service.app import app

client = TestClient(app)


def test_healthz_ok():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_tokenize_basic_korean():
    r = client.post(
        "/tokenize",
        json={"texts": ["한국주택금융공사 채용계획"], "filterPos": ["NNG", "NNP"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "tokens" in body
    assert len(body["tokens"]) == 1
    tokens = body["tokens"][0]
    # Kiwi 가 위 입력을 적어도 '한국' '주택' '금융' '공사' '채용' '계획' 중 4개 이상으로 쪼개야 한다.
    assert sum(1 for t in tokens if t in {"한국", "주택", "금융", "공사", "채용", "계획"}) >= 4


def test_tokenize_filters_unwanted_pos():
    r = client.post(
        "/tokenize",
        json={"texts": ["나는 학교에 갔다"], "filterPos": ["NNG", "NNP"]},
    )
    assert r.status_code == 200
    tokens = r.json()["tokens"][0]
    # '학교' 는 NNG. '나', '갔다' 등은 NP/VV 라 필터에서 빠져야 한다.
    assert "학교" in tokens
    assert "나" not in tokens


def test_tokenize_empty_batch():
    r = client.post("/tokenize", json={"texts": [], "filterPos": ["NNG"]})
    assert r.status_code == 200
    assert r.json() == {"tokens": []}


def test_tokenize_non_korean_passes_through_empty():
    r = client.post(
        "/tokenize",
        json={"texts": ["hello world 123"], "filterPos": ["NNG", "NNP"]},
    )
    assert r.status_code == 200
    # 한국어 명사가 없으므로 빈 토큰 배열이어야 한다 (영문/숫자는 NNG 필터에서 제외).
    assert r.json()["tokens"][0] == []
```

- [ ] **Step 1.3: Add pytest config**

`kiwi-service/pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 1.4: Run the test to verify it fails**

```bash
cd kiwi-service && python -m pytest -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'kiwi_service'`.

- [ ] **Step 1.5: Add requirements**

`kiwi-service/requirements.txt`:

```
fastapi==0.115.4
uvicorn[standard]==0.32.0
kiwipiepy==0.18.0
pydantic==2.9.2
pytest==8.3.3
httpx==0.27.2
```

```bash
pip install -r requirements.txt
```

- [ ] **Step 1.6: Implement the FastAPI app**

`kiwi-service/app.py`:

```python
"""kiwi-service: small FastAPI sidecar that tokenizes Korean text with Kiwi.

Single endpoint POST /tokenize accepts a batch of texts plus an allow-list of
part-of-speech tags, returning the tokens per text after filtering. A separate
GET /healthz is provided for container health checks.
"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI
from kiwipiepy import Kiwi
from pydantic import BaseModel, Field


class TokenizeRequest(BaseModel):
    texts: List[str] = Field(default_factory=list)
    filterPos: List[str] = Field(default_factory=lambda: ["NNG", "NNP", "SL", "SH", "SN"])


class TokenizeResponse(BaseModel):
    tokens: List[List[str]]


app = FastAPI(title="kiwi-service", version="1.0.0")
_kiwi = Kiwi()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest) -> TokenizeResponse:
    if not req.texts:
        return TokenizeResponse(tokens=[])
    allow = set(req.filterPos)
    out: List[List[str]] = []
    for analysis in _kiwi.tokenize(req.texts):
        filtered = [t.form for t in analysis if t.tag in allow]
        out.append(filtered)
    return TokenizeResponse(tokens=out)
```

To match the test import `from kiwi_service.app import app`, also create a package shim. Add `kiwi-service/kiwi_service/__init__.py` (empty) and `kiwi-service/kiwi_service/app.py`:

```python
from app import app  # re-export for `kiwi_service.app` import path
__all__ = ["app"]
```

Alternative (simpler): change the test's import to `from app import app` and skip the shim. Pick one and apply consistently. **This plan uses the shim** so the package name is unambiguous in container builds.

- [ ] **Step 1.7: Run tests to verify they pass**

```bash
cd kiwi-service && python -m pytest -v
```

Expected: 5 tests pass.

- [ ] **Step 1.8: Write the Dockerfile**

`kiwi-service/Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY kiwi_service/ ./kiwi_service/

EXPOSE 8765

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8765/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]
```

`kiwi-service/.dockerignore`:

```
tests/
__pycache__/
*.pyc
.pytest_cache/
.venv/
```

- [ ] **Step 1.9: Build the image**

```bash
docker build -t kiwi-service:dev kiwi-service/
docker run --rm -d -p 8765:8765 --name kiwi-service-test kiwi-service:dev
sleep 3
curl -fsS http://localhost:8765/healthz
echo
curl -fsS -H 'content-type: application/json' \
  -d '{"texts":["한국주택금융공사 채용계획"],"filterPos":["NNG","NNP"]}' \
  http://localhost:8765/tokenize
echo
docker stop kiwi-service-test
```

Expected: `{"status":"ok"}` then a JSON object containing a `tokens` array with at least 4 Korean tokens from the sentence.

- [ ] **Step 1.10: Add the compose profile**

In `docker/docker-compose.yml`, append (preserving existing indentation level under `services:`):

```yaml
  kiwi-service:
    container_name: kiwi-service
    profiles: ["hybrid"]
    build:
      context: ../kiwi-service
      dockerfile: Dockerfile
    networks:
      - anything-llm
    ports:
      - "8765:8765"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8765/healthz', timeout=2).status==200 else 1)\""]
      interval: 10s
      timeout: 3s
      retries: 5
```

- [ ] **Step 1.11: Smoke-test the compose profile**

```bash
cd docker && docker compose --profile hybrid up -d kiwi-service
docker compose ps kiwi-service
sleep 5
docker compose exec kiwi-service curl -fsS http://localhost:8765/healthz
docker compose --profile hybrid down
```

Expected: `kiwi-service` container `healthy`, `/healthz` returns `{"status":"ok"}`.

- [ ] **Step 1.12: Commit and PR**

```bash
git add kiwi-service/ docker/docker-compose.yml
git commit -m "feat(kiwi-service): add Korean morphological tokenizer sidecar"
git push -u origin feat/kiwi-service
gh pr create --base feature/qdrant-hybrid-kiwi --title "feat(kiwi-service): Korean morphological tokenizer sidecar" --body "FastAPI + kiwipiepy /tokenize and /healthz, Dockerfile, compose 'hybrid' profile, pytest coverage of POS filter + empty/non-Korean inputs."
```

After merge:

```bash
git checkout feature/qdrant-hybrid-kiwi && git pull --ff-only
```

---

## Task 2: Hybrid helper modules (`bm25`, `kiwiClient`, `stats`, `config`)

These are **pure** modules (or thin HTTP wrappers) wired up via TDD. They have no dependency on the Qdrant provider yet — they only deal with their own data shapes.

**Files:**
- Create: `server/utils/vectorDbProviders/qdrant/hybrid/config.js`
- Create: `server/utils/vectorDbProviders/qdrant/hybrid/bm25.js`
- Create: `server/utils/vectorDbProviders/qdrant/hybrid/kiwiClient.js`
- Create: `server/utils/vectorDbProviders/qdrant/hybrid/stats.js`
- Create: `server/utils/vectorDbProviders/qdrant/hybrid/index.js`
- Create: `server/__tests__/vectorDbProviders/qdrant/bm25.test.js`
- Create: `server/__tests__/vectorDbProviders/qdrant/kiwiClient.test.js`
- Create: `server/__tests__/vectorDbProviders/qdrant/stats.test.js`

- [ ] **Step 2.1: Branch**

```bash
git checkout feature/qdrant-hybrid-kiwi && git pull --ff-only
git checkout -b feat/qdrant-hybrid-helpers
```

- [ ] **Step 2.2: Write failing `bm25.test.js`**

`server/__tests__/vectorDbProviders/qdrant/bm25.test.js`:

```js
const {
  hashToken,
  buildDocSparse,
  buildQuerySparse,
  computeIdf,
} = require("../../../utils/vectorDbProviders/qdrant/hybrid/bm25");

describe("bm25.hashToken", () => {
  it("returns deterministic 20-bit non-negative integers", () => {
    expect(hashToken("주택")).toBe(hashToken("주택"));
    expect(hashToken("주택")).toBeGreaterThanOrEqual(0);
    expect(hashToken("주택")).toBeLessThan(1 << 20);
  });

  it("distributes distinct tokens to (usually) distinct buckets", () => {
    const a = hashToken("주택");
    const b = hashToken("금융");
    expect(a).not.toBe(b);
  });
});

describe("bm25.buildDocSparse", () => {
  it("produces (indices,values) with BM25 doc-side TF saturation and length norm", () => {
    const tokens = ["주택", "주택", "금융", "공사"];
    const out = buildDocSparse(tokens, { avgdl: 4, k1: 1.2, b: 0.75 });
    expect(out.indices.length).toBe(3);
    expect(out.values.length).toBe(out.indices.length);
    // 주택 (tf=2) 의 가중치가 금융/공사 (tf=1) 보다 커야 한다.
    const idxJutaek = out.indices.indexOf(hashToken("주택"));
    const idxKumyung = out.indices.indexOf(hashToken("금융"));
    expect(out.values[idxJutaek]).toBeGreaterThan(out.values[idxKumyung]);
  });

  it("returns empty sparse for empty token list", () => {
    expect(buildDocSparse([], { avgdl: 4, k1: 1.2, b: 0.75 })).toEqual({
      indices: [],
      values: [],
    });
  });
});

describe("bm25.computeIdf", () => {
  it("uses ln((N - df + 0.5)/(df + 0.5) + 1)", () => {
    const idf = computeIdf({ N: 100, df: 10 });
    const expected = Math.log((100 - 10 + 0.5) / (10 + 0.5) + 1);
    expect(idf).toBeCloseTo(expected, 6);
  });

  it("returns 0 when token has no df", () => {
    expect(computeIdf({ N: 100, df: 0 })).toBeCloseTo(Math.log(100 / 0.5 + 1), 6);
  });
});

describe("bm25.buildQuerySparse", () => {
  it("weights each query token by its IDF using the stats", () => {
    const tokens = ["주택", "금융"];
    const stats = { N: 100, df: { [hashToken("주택")]: 10, [hashToken("금융")]: 50 } };
    const out = buildQuerySparse(tokens, stats);
    expect(out.indices).toEqual(expect.arrayContaining([hashToken("주택"), hashToken("금융")]));
    const idxJ = out.indices.indexOf(hashToken("주택"));
    const idxK = out.indices.indexOf(hashToken("금융"));
    // 더 희귀한 (df 작은) 토큰의 IDF 가 더 커야 한다.
    expect(out.values[idxJ]).toBeGreaterThan(out.values[idxK]);
  });
});
```

- [ ] **Step 2.3: Run — expect failure**

```bash
cd server && yarn test __tests__/vectorDbProviders/qdrant/bm25.test.js
```

Expected: FAIL — `Cannot find module '.../hybrid/bm25'`.

- [ ] **Step 2.4: Implement `bm25.js`**

`server/utils/vectorDbProviders/qdrant/hybrid/bm25.js`:

```js
const SPARSE_DIM_BITS = 20;
const SPARSE_DIM_MASK = (1 << SPARSE_DIM_BITS) - 1;

function hashToken(token) {
  let h = 0x811c9dc5;
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h & SPARSE_DIM_MASK;
}

function buildDocSparse(tokens, { avgdl, k1, b }) {
  if (!tokens.length) return { indices: [], values: [] };
  const dl = tokens.length;
  const tf = new Map();
  for (const t of tokens) {
    const k = hashToken(t);
    tf.set(k, (tf.get(k) || 0) + 1);
  }
  const indices = [];
  const values = [];
  const norm = 1 - b + b * (dl / Math.max(avgdl, 1));
  for (const [idx, f] of tf.entries()) {
    indices.push(idx);
    values.push(((k1 + 1) * f) / (k1 * norm + f));
  }
  return { indices, values };
}

function computeIdf({ N, df }) {
  return Math.log(((N - df + 0.5) / (df + 0.5)) + 1);
}

function buildQuerySparse(tokens, stats) {
  if (!tokens.length) return { indices: [], values: [] };
  const N = Math.max(stats.N || 0, 1);
  const dfMap = stats.df || {};
  const buckets = new Map();
  for (const t of tokens) {
    const idx = hashToken(t);
    const df = dfMap[idx] || 0;
    const weight = computeIdf({ N, df });
    buckets.set(idx, (buckets.get(idx) || 0) + weight);
  }
  return {
    indices: [...buckets.keys()],
    values: [...buckets.values()],
  };
}

module.exports = { hashToken, buildDocSparse, buildQuerySparse, computeIdf, SPARSE_DIM_BITS };
```

- [ ] **Step 2.5: Run — expect pass**

```bash
yarn test __tests__/vectorDbProviders/qdrant/bm25.test.js
```

Expected: all bm25 tests PASS.

- [ ] **Step 2.6: Write failing `kiwiClient.test.js`**

`server/__tests__/vectorDbProviders/qdrant/kiwiClient.test.js`:

```js
const nock = require("nock");
const {
  KiwiClient,
} = require("../../../utils/vectorDbProviders/qdrant/hybrid/kiwiClient");

describe("KiwiClient", () => {
  const base = "http://kiwi-service:8765";
  let client;

  beforeEach(() => {
    nock.disableNetConnect();
    client = new KiwiClient({ baseUrl: base, healthCacheMs: 1000 });
  });

  afterEach(() => {
    nock.cleanAll();
    nock.enableNetConnect();
  });

  it("tokenize returns the kiwi-service tokens", async () => {
    nock(base).post("/tokenize").reply(200, { tokens: [["주택", "금융"]] });
    const out = await client.tokenize(["한국주택금융"], ["NNG", "NNP"]);
    expect(out).toEqual([["주택", "금융"]]);
  });

  it("isHealthy caches the result for healthCacheMs", async () => {
    const scope = nock(base).get("/healthz").reply(200, { status: "ok" });
    expect(await client.isHealthy()).toBe(true);
    // 두 번째 호출은 캐시. nock 가 한 번만 매칭되었어야 한다.
    expect(await client.isHealthy()).toBe(true);
    expect(scope.isDone()).toBe(true);
  });

  it("isHealthy returns false and never throws on network error", async () => {
    nock(base).get("/healthz").replyWithError("ECONNREFUSED");
    expect(await client.isHealthy()).toBe(false);
  });

  it("tokenize throws when service returns non-2xx", async () => {
    nock(base).post("/tokenize").reply(500, { error: "boom" });
    await expect(client.tokenize(["x"], ["NNG"])).rejects.toThrow(/kiwi/i);
  });
});
```

Also add `nock` to server devDeps:

In `server/package.json`, append to `devDependencies`: `"nock": "^13.5.5"`. Run `yarn install`.

- [ ] **Step 2.7: Run — expect failure**

```bash
yarn test __tests__/vectorDbProviders/qdrant/kiwiClient.test.js
```

Expected: FAIL — module missing.

- [ ] **Step 2.8: Implement `kiwiClient.js`**

`server/utils/vectorDbProviders/qdrant/hybrid/kiwiClient.js`:

```js
class KiwiClient {
  constructor({ baseUrl, healthCacheMs = 5000, fetchImpl = global.fetch } = {}) {
    if (!baseUrl) throw new Error("KiwiClient: baseUrl required");
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.healthCacheMs = healthCacheMs;
    this.fetchImpl = fetchImpl;
    this._healthCached = null;
    this._healthExpires = 0;
  }

  async tokenize(texts, filterPos = ["NNG", "NNP", "SL", "SH", "SN"]) {
    const res = await this.fetchImpl(`${this.baseUrl}/tokenize`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ texts, filterPos }),
    });
    if (!res.ok) throw new Error(`kiwi /tokenize failed: ${res.status}`);
    const body = await res.json();
    return body.tokens;
  }

  async isHealthy() {
    const now = Date.now();
    if (this._healthCached !== null && now < this._healthExpires) return this._healthCached;
    let ok = false;
    try {
      const res = await this.fetchImpl(`${this.baseUrl}/healthz`);
      ok = res.ok;
    } catch (_e) {
      ok = false;
    }
    this._healthCached = ok;
    this._healthExpires = now + this.healthCacheMs;
    return ok;
  }
}

module.exports = { KiwiClient };
```

- [ ] **Step 2.9: Run — expect pass**

```bash
yarn test __tests__/vectorDbProviders/qdrant/kiwiClient.test.js
```

Expected: all kiwiClient tests PASS.

- [ ] **Step 2.10: Write failing `stats.test.js`**

`server/__tests__/vectorDbProviders/qdrant/stats.test.js`:

```js
const {
  STATS_POINT_ID,
  readStats,
  applyDocsDelta,
} = require("../../../utils/vectorDbProviders/qdrant/hybrid/stats");

function makeClientStub(seedPayload = null) {
  const upserts = [];
  return {
    upserts,
    retrieve: jest.fn(async (_ns, { ids }) => {
      if (!seedPayload) return [];
      return ids.includes(STATS_POINT_ID) ? [{ id: STATS_POINT_ID, payload: seedPayload }] : [];
    }),
    upsert: jest.fn(async (_ns, body) => {
      upserts.push(body);
      return { status: "ok" };
    }),
  };
}

describe("stats.readStats", () => {
  it("returns empty defaults when point is missing", async () => {
    const client = makeClientStub(null);
    const s = await readStats(client, "ns");
    expect(s).toEqual({ N: 0, totalLen: 0, df: {} });
  });

  it("returns the stored payload when present", async () => {
    const client = makeClientStub({ N: 5, totalLen: 100, df: { 1: 2 } });
    const s = await readStats(client, "ns");
    expect(s).toEqual({ N: 5, totalLen: 100, df: { 1: 2 } });
  });
});

describe("stats.applyDocsDelta", () => {
  it("accumulates N, totalLen, and per-token df", async () => {
    const client = makeClientStub({ N: 1, totalLen: 4, df: { 7: 1 } });
    const docs = [
      { tokens: ["주택", "주택", "금융"], hashes: [7, 7, 9] },
      { tokens: ["공사"], hashes: [11] },
    ];
    await applyDocsDelta(client, "ns", docs);
    expect(client.upsert).toHaveBeenCalledTimes(1);
    const point = client.upsert.mock.calls[0][1].points[0];
    expect(point.id).toBe(STATS_POINT_ID);
    expect(point.payload.N).toBe(3);
    expect(point.payload.totalLen).toBe(4 + 3 + 1);
    // 7 은 doc1 에서 발견됨 → df += 1. 9, 11 은 신규 → df = 1.
    expect(point.payload.df["7"]).toBe(2);
    expect(point.payload.df["9"]).toBe(1);
    expect(point.payload.df["11"]).toBe(1);
  });
});
```

- [ ] **Step 2.11: Run — expect failure**

```bash
yarn test __tests__/vectorDbProviders/qdrant/stats.test.js
```

Expected: FAIL — module missing.

- [ ] **Step 2.12: Implement `stats.js`**

`server/utils/vectorDbProviders/qdrant/hybrid/stats.js`:

```js
// A single reserved point per Qdrant collection holds BM25 statistics.
// id is a fixed UUID v4 so it doesn't collide with content points
// (AnythingLLM content points use random UUIDs).
const STATS_POINT_ID = "00000000-0000-4000-8000-000000bm2500";

const _locks = new Map(); // namespace -> Promise chain

function _withLock(namespace, fn) {
  const prev = _locks.get(namespace) || Promise.resolve();
  const next = prev.catch(() => {}).then(fn);
  _locks.set(
    namespace,
    next.finally(() => {
      if (_locks.get(namespace) === next) _locks.delete(namespace);
    })
  );
  return next;
}

async function readStats(client, namespace) {
  const points = await client.retrieve(namespace, {
    ids: [STATS_POINT_ID],
    with_payload: true,
    with_vector: false,
  });
  if (!points || points.length === 0) return { N: 0, totalLen: 0, df: {} };
  const { N = 0, totalLen = 0, df = {} } = points[0].payload || {};
  return { N, totalLen, df };
}

async function _writeStats(client, namespace, stats) {
  // The reserved point still needs a vector field on hybrid collections; use a
  // tiny zero-vector. The collection schema task ensures `dense` exists.
  await client.upsert(namespace, {
    points: [
      {
        id: STATS_POINT_ID,
        payload: { __bm25_stats__: true, ...stats },
        vector: { dense: [], sparse: { indices: [], values: [] } },
      },
    ],
  });
}

async function applyDocsDelta(client, namespace, docs) {
  return _withLock(namespace, async () => {
    const stats = await readStats(client, namespace);
    for (const doc of docs) {
      stats.N += 1;
      stats.totalLen += doc.tokens.length;
      const seen = new Set();
      for (const h of doc.hashes) {
        if (seen.has(h)) continue;
        seen.add(h);
        stats.df[h] = (stats.df[h] || 0) + 1;
      }
    }
    await _writeStats(client, namespace, stats);
    return stats;
  });
}

function avgdl(stats) {
  return stats.N > 0 ? stats.totalLen / stats.N : 1;
}

module.exports = { STATS_POINT_ID, readStats, applyDocsDelta, avgdl };
```

> **Note on the zero-vector point:** Qdrant rejects upserts whose vector size doesn't match the collection schema. The schema task (Task 3) creates `dense` with the embedder dimension. `stats.js` here writes `vector: { dense: [], sparse: ... }` which Qdrant will accept ONLY if the collection allows variable-size vectors — which it does not. **Real implementation must pass the correct dense dimension.** Update `_writeStats` to accept a `denseDim` parameter and call sites must supply it from the collection metadata. The test above stubs the client so this passes; the indexing task (Task 4) will surface and fix the dim wiring.

- [ ] **Step 2.13: Run — expect pass**

```bash
yarn test __tests__/vectorDbProviders/qdrant/stats.test.js
```

Expected: stats tests PASS.

- [ ] **Step 2.14: Implement `config.js`**

`server/utils/vectorDbProviders/qdrant/hybrid/config.js`:

```js
function asBool(v) {
  return typeof v === "string" ? v.toLowerCase() === "true" : Boolean(v);
}

function hybridConfig() {
  return {
    enabled: asBool(process.env.QDRANT_HYBRID_ENABLED),
    kiwiServiceUrl: process.env.KIWI_SERVICE_URL || "http://kiwi-service:8765",
    fusion: process.env.QDRANT_HYBRID_FUSION || "rrf",
    bm25: {
      k1: Number(process.env.QDRANT_HYBRID_BM25_K1 || 1.2),
      b: Number(process.env.QDRANT_HYBRID_BM25_B || 0.75),
    },
    filterPos: ["NNG", "NNP", "SL", "SH", "SN"],
  };
}

module.exports = { hybridConfig };
```

- [ ] **Step 2.15: Implement `hybrid/index.js`** (barrel)

`server/utils/vectorDbProviders/qdrant/hybrid/index.js`:

```js
const bm25 = require("./bm25");
const { KiwiClient } = require("./kiwiClient");
const stats = require("./stats");
const { hybridConfig } = require("./config");

module.exports = { ...bm25, KiwiClient, ...stats, hybridConfig };
```

- [ ] **Step 2.16: Full helper test pass**

```bash
yarn test __tests__/vectorDbProviders/qdrant/
```

Expected: bm25 + kiwiClient + stats — all PASS.

- [ ] **Step 2.17: Commit and PR**

```bash
git add server/package.json server/yarn.lock server/utils/vectorDbProviders/qdrant/hybrid server/__tests__/vectorDbProviders/qdrant
git commit -m "feat(qdrant): add hybrid helper modules (bm25, kiwiClient, stats, config)"
git push -u origin feat/qdrant-hybrid-helpers
gh pr create --base feature/qdrant-hybrid-kiwi --title "feat(qdrant): hybrid helper modules" --body "Pure helpers + HTTP wrapper for the upcoming hybrid indexing/query paths. Includes BM25 sparse builder, IDF helper, Kiwi service client with health caching, and \`__bm25_stats__\` reserved-point I/O guarded by per-namespace mutex. No provider wiring yet."
```

After merge → return to integration branch.

---

## Task 3: Provider schema branching (collection creation + detection)

This is the smallest possible change to `qdrant/index.js` that introduces hybrid awareness **without** touching the indexing/query paths yet. Only collection creation and a `vectorSchema()` helper.

**Files:**
- Modify: `server/utils/vectorDbProviders/qdrant/index.js` (around `getOrCreateCollection`)
- Create: `server/__tests__/vectorDbProviders/qdrant/schemaBranching.test.js`

- [ ] **Step 3.1: Branch**

```bash
git checkout feature/qdrant-hybrid-kiwi && git pull --ff-only
git checkout -b feat/qdrant-hybrid-schema
```

- [ ] **Step 3.2: Read current `getOrCreateCollection`**

```bash
sed -n '135,160p' server/utils/vectorDbProviders/qdrant/index.js
```

Expected: shows the existing unnamed-dense `createCollection` call.

- [ ] **Step 3.3: Write failing test**

`server/__tests__/vectorDbProviders/qdrant/schemaBranching.test.js`:

```js
const path = require("path");

// We test the helpers in isolation by re-requiring after env mutation.
function loadProvider() {
  jest.resetModules();
  return require(path.join("..", "..", "..", "utils", "vectorDbProviders", "qdrant"));
}

function fakeClient(collectionResponse) {
  return {
    getCollection: jest.fn(async () => collectionResponse),
    createCollection: jest.fn(async () => ({ ok: true })),
    count: jest.fn(async () => ({ count: 0 })),
  };
}

describe("QDrant.vectorSchema", () => {
  afterEach(() => {
    delete process.env.QDRANT_HYBRID_ENABLED;
  });

  it("detects legacy unnamed-dense schema as 'dense'", async () => {
    const { QDrant } = loadProvider();
    const client = fakeClient({ config: { params: { vectors: { size: 1536, distance: "Cosine" } } } });
    const schema = await QDrant.vectorSchema(client, "legacy");
    expect(schema).toBe("dense");
  });

  it("detects named hybrid schema as 'hybrid'", async () => {
    const { QDrant } = loadProvider();
    const client = fakeClient({
      config: {
        params: {
          vectors: { dense: { size: 1536, distance: "Cosine" } },
          sparse_vectors: { sparse: {} },
        },
      },
    });
    const schema = await QDrant.vectorSchema(client, "new");
    expect(schema).toBe("hybrid");
  });

  it("creates a legacy unnamed-dense collection when hybrid disabled", async () => {
    const { QDrant } = loadProvider();
    const client = {
      getCollection: jest.fn().mockRejectedValueOnce(new Error("not found")).mockResolvedValue({}),
      createCollection: jest.fn(async () => ({ ok: true })),
      count: jest.fn(async () => ({ count: 0 })),
    };
    await QDrant.getOrCreateCollection(client, "ns", 1536);
    expect(client.createCollection).toHaveBeenCalledWith("ns", {
      vectors: { size: 1536, distance: "Cosine" },
    });
  });

  it("creates a named hybrid collection when QDRANT_HYBRID_ENABLED=true", async () => {
    process.env.QDRANT_HYBRID_ENABLED = "true";
    const { QDrant } = loadProvider();
    const client = {
      getCollection: jest.fn().mockRejectedValueOnce(new Error("not found")).mockResolvedValue({}),
      createCollection: jest.fn(async () => ({ ok: true })),
      count: jest.fn(async () => ({ count: 0 })),
    };
    // Force kiwi to appear healthy by injecting a stub via QDrant.__setKiwiClientForTest.
    QDrant.__setKiwiClientForTest({ isHealthy: async () => true });
    await QDrant.getOrCreateCollection(client, "ns", 1536);
    const [, body] = client.createCollection.mock.calls[0];
    expect(body.vectors).toEqual({ dense: { size: 1536, distance: "Cosine" } });
    expect(body.sparse_vectors).toEqual({ sparse: {} });
  });

  it("falls back to legacy when kiwi-service is unhealthy", async () => {
    process.env.QDRANT_HYBRID_ENABLED = "true";
    const { QDrant } = loadProvider();
    const client = {
      getCollection: jest.fn().mockRejectedValueOnce(new Error("not found")).mockResolvedValue({}),
      createCollection: jest.fn(async () => ({ ok: true })),
      count: jest.fn(async () => ({ count: 0 })),
    };
    QDrant.__setKiwiClientForTest({ isHealthy: async () => false });
    await QDrant.getOrCreateCollection(client, "ns", 1536);
    const [, body] = client.createCollection.mock.calls[0];
    expect(body.vectors).toEqual({ size: 1536, distance: "Cosine" });
    expect(body.sparse_vectors).toBeUndefined();
  });
});
```

- [ ] **Step 3.4: Run — expect failure**

```bash
yarn test __tests__/vectorDbProviders/qdrant/schemaBranching.test.js
```

Expected: FAIL — `vectorSchema is not a function` / `__setKiwiClientForTest is not a function`.

- [ ] **Step 3.5: Patch `qdrant/index.js`**

At the top of `server/utils/vectorDbProviders/qdrant/index.js`, add (right after the existing requires):

```js
const { KiwiClient, hybridConfig } = require("./hybrid");

let _kiwi = null;
function _kiwiClient() {
  if (_kiwi) return _kiwi;
  const cfg = hybridConfig();
  _kiwi = new KiwiClient({ baseUrl: cfg.kiwiServiceUrl });
  return _kiwi;
}
```

Replace `getOrCreateCollection` (around line 138) with:

```js
  async vectorSchema(client, namespace) {
    const coll = await client.getCollection(namespace).catch(() => null);
    if (!coll) return null;
    const v = coll?.config?.params?.vectors;
    if (v && typeof v === "object" && v.dense && !("size" in v)) return "hybrid";
    return "dense";
  },

  async getOrCreateCollection(client, namespace, dimensions = null) {
    if (await this.namespaceExists(client, namespace)) {
      return await client.getCollection(namespace);
    }
    if (!dimensions)
      throw new Error(
        `Qdrant:getOrCreateCollection Unable to infer vector dimension from input.`
      );

    const cfg = hybridConfig();
    const wantHybrid = cfg.enabled && (await _kiwiClient().isHealthy());

    if (wantHybrid) {
      await client.createCollection(namespace, {
        vectors: { dense: { size: dimensions, distance: "Cosine" } },
        sparse_vectors: { sparse: {} },
      });
    } else {
      if (cfg.enabled) {
        this.logger(
          "getOrCreateCollection",
          `kiwi-service unhealthy; creating legacy dense-only collection '${namespace}'.`
        );
      }
      await client.createCollection(namespace, {
        vectors: { size: dimensions, distance: "Cosine" },
      });
    }
    return await client.getCollection(namespace);
  },

  __setKiwiClientForTest(stub) {
    _kiwi = stub;
  },
```

(Leave the rest of the file untouched in this task — the indexing/query work happens in Tasks 4–5.)

- [ ] **Step 3.6: Run — expect pass**

```bash
yarn test __tests__/vectorDbProviders/qdrant/schemaBranching.test.js
```

Expected: all 5 schemaBranching tests PASS.

- [ ] **Step 3.7: Regression sweep**

```bash
yarn test
```

Expected: previously passing tests still pass (no broken imports).

- [ ] **Step 3.8: Commit and PR**

```bash
git add server/utils/vectorDbProviders/qdrant/index.js server/__tests__/vectorDbProviders/qdrant/schemaBranching.test.js
git commit -m "feat(qdrant): schema branching for hybrid collections"
git push -u origin feat/qdrant-hybrid-schema
gh pr create --base feature/qdrant-hybrid-kiwi --title "feat(qdrant): schema branching for hybrid collections" --body "Adds vectorSchema() detection and gated hybrid collection creation. Existing unnamed-dense collections continue to load through the dense path; QDRANT_HYBRID_ENABLED + kiwi healthy is required to create a named-vectors hybrid collection. No indexing/query changes yet."
```

After merge → return to integration branch.

---

## Task 4: Indexing path (`addDocumentToNamespace` hybrid integration)

**Files:**
- Modify: `server/utils/vectorDbProviders/qdrant/index.js` (`addDocumentToNamespace` body)
- Modify: `server/utils/vectorDbProviders/qdrant/hybrid/stats.js` (accept `denseDim`)
- Create: `server/__tests__/vectorDbProviders/qdrant/indexingHybrid.test.js`

- [ ] **Step 4.1: Branch**

```bash
git checkout feature/qdrant-hybrid-kiwi && git pull --ff-only
git checkout -b feat/qdrant-hybrid-index
```

- [ ] **Step 4.2: Update `stats.js` to accept `denseDim`**

In `server/utils/vectorDbProviders/qdrant/hybrid/stats.js`, change `_writeStats` and `applyDocsDelta`:

```js
async function _writeStats(client, namespace, stats, denseDim) {
  const vector = denseDim
    ? { dense: new Array(denseDim).fill(0), sparse: { indices: [0], values: [0] } }
    : { dense: [], sparse: { indices: [], values: [] } };
  await client.upsert(namespace, {
    points: [
      {
        id: STATS_POINT_ID,
        payload: { __bm25_stats__: true, ...stats },
        vector,
      },
    ],
  });
}

async function applyDocsDelta(client, namespace, docs, { denseDim } = {}) {
  return _withLock(namespace, async () => {
    const stats = await readStats(client, namespace);
    for (const doc of docs) {
      stats.N += 1;
      stats.totalLen += doc.tokens.length;
      const seen = new Set();
      for (const h of doc.hashes) {
        if (seen.has(h)) continue;
        seen.add(h);
        stats.df[h] = (stats.df[h] || 0) + 1;
      }
    }
    await _writeStats(client, namespace, stats, denseDim);
    return stats;
  });
}
```

Update the existing `stats.test.js` stat-application test to pass `{ denseDim: 4 }` and assert the upserted vector has length 4:

```js
await applyDocsDelta(client, "ns", docs, { denseDim: 4 });
const point = client.upsert.mock.calls[0][1].points[0];
expect(point.vector.dense).toHaveLength(4);
```

Run the suite:

```bash
yarn test __tests__/vectorDbProviders/qdrant/stats.test.js
```

Expected: PASS.

- [ ] **Step 4.3: Write failing indexing test**

`server/__tests__/vectorDbProviders/qdrant/indexingHybrid.test.js`:

```js
jest.mock("../../../utils/helpers", () => ({
  ...jest.requireActual("../../../utils/helpers"),
  getEmbeddingEngineSelection: () => ({
    embedderLimit: 8,
    embedTextInput: async (text) => new Array(4).fill(0).map((_, i) => i / (text.length || 1)),
    embedChunks: async (chunks) =>
      chunks.map(() => new Array(4).fill(0).map((_, i) => i / 10)),
  }),
  toChunks: (arr, n) => {
    const out = [];
    for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
    return out;
  },
}));

const { QDrant } = require("../../../utils/vectorDbProviders/qdrant");
const { STATS_POINT_ID } = require("../../../utils/vectorDbProviders/qdrant/hybrid/stats");

function hybridClientStub() {
  const upserts = [];
  return {
    upserts,
    getCollection: jest.fn(async () => ({
      config: {
        params: {
          vectors: { dense: { size: 4, distance: "Cosine" } },
          sparse_vectors: { sparse: {} },
        },
      },
    })),
    count: jest.fn(async () => ({ count: 0 })),
    retrieve: jest.fn(async () => []),
    upsert: jest.fn(async (_ns, body) => {
      upserts.push(body);
      return { status: "ok" };
    }),
  };
}

describe("QDrant.addDocumentToNamespace — hybrid", () => {
  beforeEach(() => {
    process.env.QDRANT_HYBRID_ENABLED = "true";
    QDrant.__setKiwiClientForTest({
      isHealthy: async () => true,
      tokenize: async (texts) => texts.map(() => ["주택", "금융", "공사"]),
    });
    QDrant.__setQdrantClientForTest(hybridClientStub());
  });

  it("upserts points with named vectors {dense, sparse}", async () => {
    const client = hybridClientStub();
    QDrant.__setQdrantClientForTest(client);

    const result = await QDrant.addDocumentToNamespace(
      "ns",
      {
        docId: "doc-1",
        pageContent: "한국주택금융공사 채용계획 본문 ...",
        metadata: { title: "t" },
      },
      "/tmp/cache/path"
    );

    expect(result?.vectorized).toBe(true);
    const contentUpserts = client.upsert.mock.calls.filter(
      ([_, body]) => body.points[0].id !== STATS_POINT_ID
    );
    expect(contentUpserts.length).toBeGreaterThan(0);
    const point = contentUpserts[0][1].points[0];
    expect(point.vector).toEqual(
      expect.objectContaining({
        dense: expect.any(Array),
        sparse: expect.objectContaining({ indices: expect.any(Array), values: expect.any(Array) }),
      })
    );
  });

  it("updates the __bm25_stats__ reserved point with N>=1", async () => {
    const client = hybridClientStub();
    QDrant.__setQdrantClientForTest(client);

    await QDrant.addDocumentToNamespace(
      "ns",
      { docId: "doc-1", pageContent: "한국주택 금융", metadata: {} },
      "/tmp/x"
    );

    const statsUpserts = client.upsert.mock.calls.filter(
      ([_, body]) => body.points[0].id === STATS_POINT_ID
    );
    expect(statsUpserts.length).toBe(1);
    expect(statsUpserts[0][1].points[0].payload.N).toBeGreaterThanOrEqual(1);
  });

  it("falls back to legacy unnamed-dense upsert on a legacy collection", async () => {
    const legacyClient = {
      ...hybridClientStub(),
      getCollection: jest.fn(async () => ({
        config: { params: { vectors: { size: 4, distance: "Cosine" } } },
      })),
    };
    QDrant.__setQdrantClientForTest(legacyClient);

    await QDrant.addDocumentToNamespace(
      "ns",
      { docId: "doc-1", pageContent: "X", metadata: {} },
      "/tmp/y"
    );

    const point = legacyClient.upsert.mock.calls[0][1].points[0];
    // legacy: unnamed vector, raw array
    expect(Array.isArray(point.vector)).toBe(true);
  });
});
```

- [ ] **Step 4.4: Run — expect failure**

```bash
yarn test __tests__/vectorDbProviders/qdrant/indexingHybrid.test.js
```

Expected: FAIL — `__setQdrantClientForTest is not a function`, or stub Qdrant client not used by provider.

- [ ] **Step 4.5: Add `__setQdrantClientForTest` injection**

Near the top of `qdrant/index.js` where `qdrantClient()` is defined, factor it out:

```js
let _injectedClient = null;
async function qdrantClient() {
  if (_injectedClient) return _injectedClient;
  // ... existing body that builds the @qdrant/js-client-rest QdrantClient ...
}
```

Add to the exported object:

```js
__setQdrantClientForTest(c) { _injectedClient = c; },
```

(Mirror the `__setKiwiClientForTest` pattern.)

- [ ] **Step 4.6: Patch `addDocumentToNamespace` to branch on schema**

Inside `addDocumentToNamespace`, after the existing chunking + embedding loop but **before** the `client.upsert(...)` call, branch:

```js
// Resolve once per call: dim + schema + (maybe) tokens.
const denseDim = vectorValues[0]?.length ?? 0;
const schema = await this.vectorSchema(client, namespace);
const useHybrid = schema === "hybrid";

let tokenizedChunks = null;
if (useHybrid) {
  const cfg = hybridConfig();
  tokenizedChunks = await _kiwiClient().tokenize(
    textChunks.map((c) => c.pageContent || c),
    cfg.filterPos
  );
}

const submission = textChunks.map((chunk, i) => {
  const id = uuidv4();
  const denseVec = vectorValues[i];

  if (!useHybrid) {
    return {
      id,
      vector: denseVec,
      payload: { ...(chunk.metadata || metadata), text: chunk.pageContent || chunk },
    };
  }

  const tokens = tokenizedChunks[i] || [];
  const hashes = tokens.map(hashToken);
  return {
    id,
    vector: {
      dense: denseVec,
      sparse: buildDocSparse(tokens, {
        avgdl: 1, // First-pass; stats.applyDocsDelta will refresh avgdl in real terms.
        k1: hybridConfig().bm25.k1,
        b: hybridConfig().bm25.b,
      }),
    },
    payload: {
      ...(chunk.metadata || metadata),
      text: chunk.pageContent || chunk,
      _tokenCount: tokens.length,
    },
    __tokens: tokens,
    __hashes: hashes,
  };
});

await client.upsert(namespace, {
  points: submission.map(({ __tokens, __hashes, ...p }) => p),
});

if (useHybrid) {
  const { applyDocsDelta } = require("./hybrid/stats");
  await applyDocsDelta(
    client,
    namespace,
    submission.map((p) => ({ tokens: p.__tokens, hashes: p.__hashes })),
    { denseDim }
  );
}
```

Add the missing requires near the top of the file:

```js
const { v4: uuidv4 } = require("uuid");
const { hashToken, buildDocSparse } = require("./hybrid/bm25");
```

(If `uuid` is already required as `uuidv4` elsewhere in the file, skip the duplicate.)

- [ ] **Step 4.7: Run — expect pass**

```bash
yarn test __tests__/vectorDbProviders/qdrant/indexingHybrid.test.js
```

Expected: all 3 indexing tests PASS.

- [ ] **Step 4.8: Full regression**

```bash
yarn test
```

Expected: nothing previously green has gone red.

- [ ] **Step 4.9: Commit and PR**

```bash
git add server/utils/vectorDbProviders/qdrant server/__tests__/vectorDbProviders/qdrant/indexingHybrid.test.js server/__tests__/vectorDbProviders/qdrant/stats.test.js
git commit -m "feat(qdrant): hybrid indexing path with BM25 sparse + stats point"
git push -u origin feat/qdrant-hybrid-index
gh pr create --base feature/qdrant-hybrid-kiwi --title "feat(qdrant): hybrid indexing path (sparse + stats)" --body "addDocumentToNamespace now branches on collection schema: hybrid collections receive named-vector points (dense + BM25 sparse hashed at 2^20) and update the reserved __bm25_stats__ point. Legacy unnamed-dense collections continue unchanged."
```

After merge → return to integration branch.

---

## Task 5: Query path (RRF fusion + dense fallback)

**Files:**
- Modify: `server/utils/vectorDbProviders/qdrant/index.js` (`similarityResponse` + `performSimilaritySearch`)
- Create: `server/__tests__/vectorDbProviders/qdrant/queryFusion.test.js`

- [ ] **Step 5.1: Branch**

```bash
git checkout feature/qdrant-hybrid-kiwi && git pull --ff-only
git checkout -b feat/qdrant-hybrid-query
```

- [ ] **Step 5.2: Write failing query test**

`server/__tests__/vectorDbProviders/qdrant/queryFusion.test.js`:

```js
const { QDrant } = require("../../../utils/vectorDbProviders/qdrant");

function fakeHits(count) {
  return Array.from({ length: count }).map((_, i) => ({
    id: `c-${i}`,
    score: 0.9 - i * 0.05,
    payload: { text: `chunk ${i}`, title: "t" },
  }));
}

describe("QDrant.similarityResponse — hybrid path", () => {
  beforeEach(() => {
    process.env.QDRANT_HYBRID_ENABLED = "true";
    QDrant.__setKiwiClientForTest({
      isHealthy: async () => true,
      tokenize: async () => [["주택", "금융"]],
    });
  });

  it("uses client.query with prefetch[dense, sparse] + fusion=rrf on hybrid collections", async () => {
    const client = {
      getCollection: jest.fn(async () => ({
        config: {
          params: {
            vectors: { dense: { size: 4, distance: "Cosine" } },
            sparse_vectors: { sparse: {} },
          },
        },
      })),
      retrieve: jest.fn(async () => [{ payload: { N: 2, totalLen: 4, df: {} } }]),
      query: jest.fn(async () => ({ points: fakeHits(3) })),
      search: jest.fn(),
    };

    const res = await QDrant.similarityResponse({
      client,
      namespace: "ns",
      queryVector: [0.1, 0.2, 0.3, 0.4],
      queryText: "주택 금융 채용",
      topN: 3,
    });

    expect(client.query).toHaveBeenCalledTimes(1);
    const [, body] = client.query.mock.calls[0];
    expect(body.prefetch).toEqual([
      expect.objectContaining({ using: "dense" }),
      expect.objectContaining({ using: "sparse" }),
    ]);
    expect(body.query.fusion).toBe("rrf");
    expect(res.contextTexts).toHaveLength(3);
    expect(client.search).not.toHaveBeenCalled();
  });

  it("falls back to dense-only client.search on legacy collections", async () => {
    const client = {
      getCollection: jest.fn(async () => ({
        config: { params: { vectors: { size: 4, distance: "Cosine" } } },
      })),
      query: jest.fn(),
      search: jest.fn(async () => fakeHits(2)),
    };

    const res = await QDrant.similarityResponse({
      client,
      namespace: "ns",
      queryVector: [0.1, 0.2, 0.3, 0.4],
      queryText: "주택 금융",
      topN: 2,
    });

    expect(client.search).toHaveBeenCalledTimes(1);
    expect(client.query).not.toHaveBeenCalled();
    expect(res.contextTexts).toHaveLength(2);
  });

  it("falls back to dense-only when kiwi is unhealthy on a hybrid collection", async () => {
    QDrant.__setKiwiClientForTest({ isHealthy: async () => false });
    const client = {
      getCollection: jest.fn(async () => ({
        config: {
          params: {
            vectors: { dense: { size: 4, distance: "Cosine" } },
            sparse_vectors: { sparse: {} },
          },
        },
      })),
      retrieve: jest.fn(async () => [{ payload: { N: 1, totalLen: 2, df: {} } }]),
      query: jest.fn(async () => ({ points: fakeHits(2) })),
      search: jest.fn(),
    };

    await QDrant.similarityResponse({
      client,
      namespace: "ns",
      queryVector: [0.1, 0.2, 0.3, 0.4],
      queryText: "X",
      topN: 2,
    });

    // hybrid collection 이지만 kiwi 불가 → dense-only prefetch (sparse 없이 query)
    const [, body] = client.query.mock.calls[0];
    expect(body.prefetch.length).toBe(1);
    expect(body.prefetch[0].using).toBe("dense");
  });
});
```

- [ ] **Step 5.3: Run — expect failure**

```bash
yarn test __tests__/vectorDbProviders/qdrant/queryFusion.test.js
```

Expected: FAIL — `client.query is not used`.

- [ ] **Step 5.4: Patch `similarityResponse`**

Replace the existing `similarityResponse` (around line 62) with:

```js
async similarityResponse({
  client,
  namespace,
  queryVector,
  queryText = "",
  similarityThreshold = 0.25,
  topN = 4,
  filterIdentifiers = [],
}) {
  const result = { contextTexts: [], sourceDocuments: [], scores: [] };

  const schema = await this.vectorSchema(client, namespace);
  const isHybridColl = schema === "hybrid";

  const kiwi = isHybridColl ? _kiwiClient() : null;
  const kiwiHealthy = kiwi ? await kiwi.isHealthy() : false;

  let hits = [];
  if (!isHybridColl) {
    hits = await client.search(namespace, {
      vector: queryVector,
      limit: topN,
      with_payload: true,
      score_threshold: similarityThreshold,
    });
  } else {
    const prefetch = [
      {
        using: "dense",
        query: queryVector,
        limit: Math.max(topN * 10, 50),
      },
    ];

    if (kiwiHealthy && queryText) {
      const { buildQuerySparse, readStats } = require("./hybrid");
      const cfg = hybridConfig();
      const tokens = (await kiwi.tokenize([queryText], cfg.filterPos))[0] || [];
      if (tokens.length) {
        const stats = await readStats(client, namespace);
        const sparseQuery = buildQuerySparse(tokens, stats);
        if (sparseQuery.indices.length) {
          prefetch.push({
            using: "sparse",
            query: sparseQuery,
            limit: Math.max(topN * 10, 50),
          });
        }
      }
    } else if (isHybridColl) {
      this.logger(
        "similarityResponse",
        `kiwi unhealthy or empty query; using dense-only prefetch on hybrid collection '${namespace}'.`
      );
    }

    const response = await client.query(namespace, {
      prefetch,
      query: { fusion: hybridConfig().fusion },
      limit: topN,
      with_payload: true,
      score_threshold: similarityThreshold,
    });
    hits = response?.points || [];
  }

  for (const h of hits) {
    if (!h?.payload?.text) continue;
    if (filterIdentifiers.includes(h.payload?.docId)) continue;
    result.contextTexts.push(h.payload.text);
    result.sourceDocuments.push(h.payload);
    result.scores.push(h.score);
  }

  return result;
},
```

- [ ] **Step 5.5: Update `performSimilaritySearch` to forward `queryText`**

Find `performSimilaritySearch` (around line 351). Locate the existing `similarityResponse({ ... })` call inside it and add `queryText` to the call site:

```js
const result = await this.similarityResponse({
  client,
  namespace,
  queryVector,
  queryText: input,        // <-- ADD: pass the original user query through.
  similarityThreshold,
  topN,
  filterIdentifiers,
});
```

(Variable name for the user query may differ — confirm with `grep -n "async performSimilaritySearch" server/utils/vectorDbProviders/qdrant/index.js` and use whatever name the existing method signature uses for the raw text query.)

- [ ] **Step 5.6: Run — expect pass**

```bash
yarn test __tests__/vectorDbProviders/qdrant/queryFusion.test.js
```

Expected: all 3 query tests PASS.

- [ ] **Step 5.7: Full regression**

```bash
yarn test
```

Expected: green.

- [ ] **Step 5.8: Commit and PR**

```bash
git add server/utils/vectorDbProviders/qdrant/index.js server/__tests__/vectorDbProviders/qdrant/queryFusion.test.js
git commit -m "feat(qdrant): hybrid query path with RRF fusion + dense fallback"
git push -u origin feat/qdrant-hybrid-query
gh pr create --base feature/qdrant-hybrid-kiwi --title "feat(qdrant): hybrid query path (RRF + fallbacks)" --body "similarityResponse uses Qdrant Query API with prefetch[dense, sparse] + RRF fusion on hybrid collections, dense-only client.search on legacy collections, and dense-only prefetch when kiwi-service is unreachable. performSimilaritySearch forwards the raw user query for tokenization."
```

After merge → return to integration branch.

---

## Task 6: End-to-end integration test with the sample PDF

**Files:**
- Create: `server/__tests__/integration/qdrantHybridE2E.test.js`
- Create: `scripts/integration/start-hybrid-stack.sh`
- Modify: `samples/qdrant-hybrid-kiwi/README.md` (note about e2e)

- [ ] **Step 6.1: Branch**

```bash
git checkout feature/qdrant-hybrid-kiwi && git pull --ff-only
git checkout -b feat/qdrant-hybrid-e2e
```

- [ ] **Step 6.2: Add the stack helper**

`scripts/integration/start-hybrid-stack.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROFILE_FLAG="--profile hybrid"
COMPOSE_DIR="$(cd "$(dirname "$0")/../.." && pwd)/docker"
QDRANT_PORT="${QDRANT_PORT:-6333}"
KIWI_PORT="${KIWI_PORT:-8765}"

cmd="${1:-up}"

case "$cmd" in
  up)
    docker run -d --rm --name qdrant-e2e -p ${QDRANT_PORT}:6333 qdrant/qdrant:v1.12.4
    (cd "$COMPOSE_DIR" && docker compose ${PROFILE_FLAG} up -d kiwi-service)
    # wait
    for i in $(seq 1 30); do
      if curl -fsS "http://localhost:${KIWI_PORT}/healthz" >/dev/null \
         && curl -fsS "http://localhost:${QDRANT_PORT}/readyz" >/dev/null; then
        echo "stack ready"
        exit 0
      fi
      sleep 1
    done
    echo "stack failed to become ready" >&2
    exit 1
    ;;
  down)
    docker stop qdrant-e2e >/dev/null 2>&1 || true
    (cd "$COMPOSE_DIR" && docker compose ${PROFILE_FLAG} down) || true
    ;;
  *) echo "usage: $0 {up|down}" >&2; exit 2;;
esac
```

```bash
chmod +x scripts/integration/start-hybrid-stack.sh
```

- [ ] **Step 6.3: Write the e2e test**

`server/__tests__/integration/qdrantHybridE2E.test.js`:

```js
/**
 * End-to-end integration test for hybrid retrieval against a live local stack.
 *
 * Requires:
 *   - Running Qdrant at $QDRANT_ENDPOINT (default http://localhost:6333)
 *   - Running kiwi-service at $KIWI_SERVICE_URL (default http://localhost:8765)
 *   - Sample PDF at samples/qdrant-hybrid-kiwi/raw/2026_hf_recruitment.pdf
 *
 * The harness ingests the PDF text twice — once with hybrid OFF (baseline) and
 * once with hybrid ON — into separate namespaces, then runs the fixture
 * queries and asserts:
 *   hybrid_hit_rate >= baseline_hit_rate
 */
const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");

const SAMPLE = path.resolve(
  __dirname,
  "../../../samples/qdrant-hybrid-kiwi/raw/2026_hf_recruitment.pdf"
);
const FIXTURES = path.resolve(
  __dirname,
  "../../../samples/qdrant-hybrid-kiwi/fixtures/expected_queries.json"
);

const skip = !fs.existsSync(SAMPLE) || process.env.RUN_E2E !== "1";
const d = skip ? describe.skip : describe;

d("Qdrant hybrid e2e", () => {
  jest.setTimeout(180000);

  const { QDrant } = require("../../utils/vectorDbProviders/qdrant");

  async function ingest(namespace, hybrid) {
    const prev = process.env.QDRANT_HYBRID_ENABLED;
    process.env.QDRANT_HYBRID_ENABLED = hybrid ? "true" : "false";
    try {
      const buf = fs.readFileSync(SAMPLE);
      const parsed = await pdfParse(buf);
      const text = parsed.text.replace(/\s+/g, " ");
      const chunks = text.match(/.{1,800}/g) || [];
      for (let i = 0; i < chunks.length; i++) {
        await QDrant.addDocumentToNamespace(
          namespace,
          { docId: `doc-${i}`, pageContent: chunks[i], metadata: { source: SAMPLE, chunk: i } },
          path.join("/tmp", `${namespace}-${i}.json`)
        );
      }
    } finally {
      process.env.QDRANT_HYBRID_ENABLED = prev;
    }
  }

  async function hitRate(namespace) {
    const fixtures = JSON.parse(fs.readFileSync(FIXTURES, "utf8"));
    const topK = fixtures._meta.topK;
    let hits = 0;
    let total = 0;
    for (const q of fixtures.queries) {
      const { client } = await require("../../utils/vectorDbProviders/qdrant")
        .QDrant.connect();
      const res = await QDrant.similarityResponse({
        client,
        namespace,
        queryVector: await require("../../utils/helpers")
          .getEmbeddingEngineSelection()
          .embedTextInput(q.query),
        queryText: q.query,
        topN: topK,
      });
      const blob = res.contextTexts.join(" ");
      for (const kw of q.expected_keywords) {
        total += 1;
        if (blob.includes(kw)) hits += 1;
      }
    }
    return hits / total;
  }

  const nsBase = "e2e-baseline-" + Date.now();
  const nsHyb = "e2e-hybrid-" + Date.now();

  it("hybrid hit-rate >= baseline hit-rate", async () => {
    await ingest(nsBase, false);
    await ingest(nsHyb, true);
    const base = await hitRate(nsBase);
    const hyb = await hitRate(nsHyb);
    // eslint-disable-next-line no-console
    console.log(`baseline=${base.toFixed(3)}  hybrid=${hyb.toFixed(3)}`);
    expect(hyb).toBeGreaterThanOrEqual(base);
  });
});
```

Add `pdf-parse` to server devDeps if not present:

```bash
cd server && yarn add -D pdf-parse@^1.1.1
```

- [ ] **Step 6.4: Bring up the stack and run**

```bash
./scripts/integration/start-hybrid-stack.sh up
cd server && RUN_E2E=1 \
  QDRANT_ENDPOINT=http://localhost:6333 \
  KIWI_SERVICE_URL=http://localhost:8765 \
  QDRANT_HYBRID_ENABLED=true \
  yarn test:integration __tests__/integration/qdrantHybridE2E.test.js
./scripts/integration/start-hybrid-stack.sh down
```

Expected: `hybrid hit-rate >= baseline hit-rate` PASS. The console line prints both values.

- [ ] **Step 6.5: Commit and PR**

```bash
git add server/__tests__/integration scripts/integration samples/qdrant-hybrid-kiwi/README.md server/package.json server/yarn.lock
git commit -m "test(qdrant): hybrid e2e with sample Korean PDF fixture"
git push -u origin feat/qdrant-hybrid-e2e
gh pr create --base feature/qdrant-hybrid-kiwi --title "test(qdrant): hybrid e2e with sample Korean PDF" --body "End-to-end gated by RUN_E2E=1. Spins up Qdrant + kiwi-service, ingests the 2026 HF PDF twice (dense baseline vs hybrid), and asserts hybrid hit-rate over the fixture queries is at least the baseline. Skips automatically when the sample PDF or RUN_E2E flag are absent."
```

---

## Task 7: Final integration + upstream PR

- [ ] **Step 7.1: Confirm all sub-PRs are merged into `feature/qdrant-hybrid-kiwi`**

```bash
git checkout feature/qdrant-hybrid-kiwi && git pull --ff-only
gh pr list --base feature/qdrant-hybrid-kiwi --state open
```

Expected: no open PRs against the integration branch.

- [ ] **Step 7.2: Re-run the full server test suite**

```bash
cd server && yarn test
```

Expected: green.

- [ ] **Step 7.3: Re-run the e2e once more from a clean stack**

```bash
./scripts/integration/start-hybrid-stack.sh down || true
./scripts/integration/start-hybrid-stack.sh up
cd server && RUN_E2E=1 \
  QDRANT_ENDPOINT=http://localhost:6333 \
  KIWI_SERVICE_URL=http://localhost:8765 \
  QDRANT_HYBRID_ENABLED=true \
  yarn test:integration
./scripts/integration/start-hybrid-stack.sh down
```

Expected: e2e PASS.

- [ ] **Step 7.4: Manual smoke (widget)**

1. `cd docker && docker compose --profile hybrid up -d`
2. Launch the AnythingLLM server locally with the new env vars set.
3. Create a fresh workspace; upload `samples/qdrant-hybrid-kiwi/raw/2026_hf_recruitment.pdf`.
4. Open the embedded chat widget for that workspace; ask "전형직 채용 인원이 몇 명인가요?".
5. Verify the response cites chunks from the PDF and the UI is unchanged.

- [ ] **Step 7.5: Upstream PR**

```bash
gh pr create --base master --title "feat: Qdrant hybrid (BM25 sparse + dense) with Kiwi tokenizer for Korean" \
  --body "$(cat <<'EOF'
## Summary
- Adds optional hybrid retrieval for the Qdrant vector-DB provider using Qdrant's named vectors (`dense` + `sparse`) and the Query API with RRF fusion.
- Introduces a small Python sidecar `kiwi-service` (FastAPI + kiwipiepy) so indexing and querying share an identical Korean morphological tokenization.
- Gated by `QDRANT_HYBRID_ENABLED` + `KIWI_SERVICE_URL` env vars and a Docker Compose `hybrid` profile. Existing dense-only collections are untouched and continue to work via automatic fallback.

## Test plan
- [ ] `yarn test` in `server/` is green (unit tests for bm25, kiwi client, stats, schema branching, indexing, query fusion).
- [ ] `pytest` in `kiwi-service/` is green.
- [ ] `RUN_E2E=1 yarn test:integration` against a local Qdrant + kiwi-service stack passes (hybrid hit-rate >= dense-only baseline on the sample Korean PDF).
- [ ] Manual: legacy collection with hybrid disabled — no behavior change.
- [ ] Manual: legacy collection with hybrid enabled — auto-fallback log appears once, search still works.
- [ ] Manual: new collection with hybrid enabled — created as named vectors, hybrid path used.
EOF
)"
```

---

## Self-Review

**Spec coverage**
- Components — new (`kiwi-service`, compose, helpers): Task 1, Task 2 ✓
- Components — changed (qdrant provider, env, helpers): Tasks 0, 3, 4, 5 ✓
- Indexing data flow (1–6): Task 4 steps 4.6 (chunks, dense, tokenize, sparse, stats upsert, named-vector upsert) ✓
- Query data flow (1–5): Task 5 step 5.4 ✓
- Backward compat (legacy unnamed dense): Task 3 (creation) + Task 4 (legacy upsert) + Task 5 (search fallback) ✓
- Error handling — kiwi unreachable: Task 5 dense-only fallback test + Task 3 unhealthy fallback test ✓
- Error handling — empty / non-Korean query: Task 5 step 5.4 (`if (tokens.length)`) ✓
- Error handling — Qdrant client version: Task 0 bumps client to ^1.10.0 ✓
- Sample data folder: already created on master; Task 6 wires it into the e2e ✓
- Testing strategy (unit, integration, regression, smoke): Tasks 1–6 + 7.4 ✓
- Git strategy (long-lived integration branch + sub-PRs): Tasks 0–7 ✓
- Sub-agent decomposition: each task maps 1:1 to a sub-agent in the spec — Task 1=kiwi-service-agent, Task 2/3=qdrant-schema-agent (+helpers), Task 4=server-indexing-agent, Task 5=server-query-agent, Task 6=integration-test-agent ✓
- Acceptance criteria 1–5: covered by unit + e2e tests + Task 7.4 ✓

**Placeholder scan**: none found.

**Type consistency**: `vectorSchema`, `__setKiwiClientForTest`, `__setQdrantClientForTest`, `STATS_POINT_ID`, `applyDocsDelta({ denseDim })`, `similarityResponse({ queryText })` all used consistently across tasks.
