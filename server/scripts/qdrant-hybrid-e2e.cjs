#!/usr/bin/env node
/**
 * End-to-end check for the Qdrant hybrid retrieval path.
 *
 * Runs as a PLAIN NODE script (not Jest) on purpose: the native embedder
 * loads `@xenova/transformers` via dynamic ESM import and creates onnxruntime
 * tensors, neither of which survive Jest's per-realm VM sandbox. Production
 * runs the embedder under plain node too, so this matches real behavior and
 * keeps full embedding fidelity.
 *
 * Requires (otherwise it prints a reason and exits 0 so it no-ops in CI):
 *   - RUN_E2E=1 in env
 *   - Sample PDF at samples/qdrant-hybrid-kiwi/raw/2026_hf_recruitment.pdf
 *   - Fixtures at samples/qdrant-hybrid-kiwi/fixtures/expected_queries.json
 * Requires live services (exits 1 if RUN_E2E=1 but they are unreachable):
 *   - Live Qdrant at QDRANT_ENDPOINT (default http://localhost:6333)
 *   - Live kiwi-service at KIWI_SERVICE_URL (default http://localhost:8765)
 *
 * Ingests the PDF into two namespaces (dense baseline vs hybrid), runs the
 * fixture queries against each, and asserts hybrid_hit_rate >= baseline.
 *
 * Boot with:
 *   ./scripts/integration/start-hybrid-stack.sh up
 *   RUN_E2E=1 \
 *     QDRANT_ENDPOINT=http://localhost:6333 \
 *     KIWI_SERVICE_URL=http://localhost:8765 \
 *     QDRANT_HYBRID_ENABLED=true \
 *     yarn test:e2e
 *   ./scripts/integration/start-hybrid-stack.sh down
 */
process.env.STORAGE_DIR = process.env.STORAGE_DIR || "/tmp/anythingllm-test";
process.env.VECTOR_DB = "qdrant";

const fs = require("fs");
const path = require("path");
const http = require("http");

const SAMPLE = path.resolve(
  __dirname,
  "../../samples/qdrant-hybrid-kiwi/raw/2026_hf_recruitment.pdf"
);
const FIXTURES = path.resolve(
  __dirname,
  "../../samples/qdrant-hybrid-kiwi/fixtures/expected_queries.json"
);
const QDRANT_ENDPOINT = process.env.QDRANT_ENDPOINT || "http://localhost:6333";
const KIWI_URL = process.env.KIWI_SERVICE_URL || "http://localhost:8765";

function reachable(url) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      res.resume();
      resolve(res.statusCode && res.statusCode < 500);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(2000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function ingest({ QDrant, pdfParse }, namespace, hybrid) {
  const prev = process.env.QDRANT_HYBRID_ENABLED;
  process.env.QDRANT_HYBRID_ENABLED = hybrid ? "true" : "false";
  try {
    const buf = fs.readFileSync(SAMPLE);
    const parsed = await pdfParse(buf);
    const text = parsed.text.replace(/\s+/g, " ");
    const chunks = text.match(/.{1,800}/g) || [];
    const db = new QDrant();
    for (let i = 0; i < chunks.length; i++) {
      await db.addDocumentToNamespace(
        namespace,
        {
          docId: `doc-${i}`,
          pageContent: chunks[i],
          metadata: { source: SAMPLE, chunk: i },
        },
        path.join("/tmp", `${namespace}-${i}.json`),
        true // skipCache
      );
    }
  } finally {
    process.env.QDRANT_HYBRID_ENABLED = prev;
  }
}

async function hitRate({ QDrant, helpers }, namespace) {
  const fixtures = JSON.parse(fs.readFileSync(FIXTURES, "utf8"));
  const topK = fixtures._meta.topK || 5;
  const db = new QDrant();
  const { client } = await db.connect();
  let hits = 0;
  let total = 0;
  for (const q of fixtures.queries) {
    const embedder = helpers.getEmbeddingEngineSelection();
    const queryVector = await embedder.embedTextInput(q.query);
    const res = await db.similarityResponse({
      client,
      namespace,
      queryVector,
      queryText: q.query,
      topN: topK,
    });
    const blob = res.contextTexts.join(" ");
    for (const kw of q.expected_keywords) {
      total += 1;
      if (blob.includes(kw)) hits += 1;
    }
  }
  return total > 0 ? hits / total : 0;
}

async function main() {
  if (process.env.RUN_E2E !== "1") {
    console.log("[e2e] skipped: RUN_E2E != 1");
    return 0;
  }
  if (!fs.existsSync(SAMPLE)) {
    console.log(`[e2e] skipped: sample PDF missing at ${SAMPLE}`);
    return 0;
  }
  if (!fs.existsSync(FIXTURES)) {
    console.log(`[e2e] skipped: fixtures missing at ${FIXTURES}`);
    return 0;
  }

  const qOk = await reachable(`${QDRANT_ENDPOINT}/readyz`);
  const kOk = await reachable(`${KIWI_URL}/healthz`);
  if (!qOk || !kOk) {
    console.error(
      `[e2e] services not reachable: qdrant=${qOk} kiwi=${kOk} ` +
        `(QDRANT_ENDPOINT=${QDRANT_ENDPOINT}, KIWI_SERVICE_URL=${KIWI_URL})`
    );
    return 1;
  }

  // NativeEmbedder does a non-recursive mkdir of STORAGE_DIR/models, which
  // assumes STORAGE_DIR already exists (true in real installs). The e2e
  // points STORAGE_DIR at a throwaway tmp path, so create the tree up front.
  fs.mkdirSync(path.join(process.env.STORAGE_DIR, "models"), {
    recursive: true,
  });

  const helpers = require("../utils/helpers");
  const { QDrant } = require("../utils/vectorDbProviders/qdrant");
  const pdfParse = require("pdf-parse");
  const deps = { QDrant, helpers, pdfParse };

  const nsBase = "e2e-baseline-" + Date.now();
  const nsHyb = "e2e-hybrid-" + Date.now();

  await ingest(deps, nsBase, false);
  await ingest(deps, nsHyb, true);
  const base = await hitRate(deps, nsBase);
  const hyb = await hitRate(deps, nsHyb);
  console.log(
    `e2e:hit-rates baseline=${base.toFixed(3)} hybrid=${hyb.toFixed(3)}`
  );

  if (hyb >= base) {
    console.log("[e2e] PASS: hybrid hit-rate >= baseline");
    return 0;
  }
  console.error("[e2e] FAIL: hybrid hit-rate < baseline");
  return 1;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error("[e2e] error:", err && err.stack ? err.stack : err);
    process.exit(1);
  });
