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

The integration test (added under `feat/qdrant-hybrid-sample-e2e`) ingests
this PDF through the collector into a fresh Qdrant workspace, then runs each
query in `fixtures/expected_queries.json` and asserts that the expected
keywords appear in the top-K results with hit-rate ≥ dense-only baseline.
