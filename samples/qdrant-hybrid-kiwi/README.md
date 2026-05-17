# Sample data — Qdrant hybrid + Kiwi (Korean)

End-to-end test fixture for the Qdrant hybrid search + Kiwi morphological
tokenizer feature.

## Contents

- `raw/2026_hf_recruitment.pdf` — 2026년도 한국주택금융공사 채용계획.
  Original filename: `2026년도 한국주택금융공사 채용계획.pdf`.
  Source: 한국주택금융공사 공개 게재 자료.
  **Not committed to git** (see repo `.gitignore`). Place a copy here locally.
- `fixtures/expected_queries.json` — Korean queries with expected top-K
  keywords, used by the integration test to compare hybrid vs dense-only
  retrieval quality.

## Obtaining the PDF

```bash
# Linux/WSL example — adjust source path for your machine.
cp "/mnt/c/Users/<you>/Downloads/2026년도 한국주택금융공사 채용계획.pdf" \
   samples/qdrant-hybrid-kiwi/raw/2026_hf_recruitment.pdf
```

## Using the fixture

The e2e runner (`server/scripts/qdrant-hybrid-e2e.cjs`) ingests this PDF into
two fresh Qdrant namespaces — one dense-only baseline, one hybrid — then runs
each query in `fixtures/expected_queries.json` and asserts that the expected
keywords appear in the top-K results with hybrid hit-rate ≥ dense-only
baseline. It runs as a plain Node script (not Jest) because the native
embedder's dynamic ESM import and onnxruntime tensors do not survive Jest's
per-realm VM sandbox.

## Running the e2e

```bash
./scripts/integration/start-hybrid-stack.sh up
cd server && RUN_E2E=1 \
  QDRANT_ENDPOINT=http://localhost:6333 \
  KIWI_SERVICE_URL=http://localhost:8765 \
  QDRANT_HYBRID_ENABLED=true \
  yarn test:e2e
./scripts/integration/start-hybrid-stack.sh down
```

The runner skips (exit 0) when `RUN_E2E` is unset or the sample PDF / fixtures
are missing, and exits 1 if the services are unreachable or hybrid regresses.
