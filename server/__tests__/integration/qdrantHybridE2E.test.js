/**
 * End-to-end integration test for the Qdrant hybrid retrieval path.
 *
 * Requires (all three; otherwise the suite auto-skips):
 *   - RUN_E2E=1 in env
 *   - Sample PDF at samples/qdrant-hybrid-kiwi/raw/2026_hf_recruitment.pdf
 *   - Live Qdrant at QDRANT_ENDPOINT (default http://localhost:6333)
 *   - Live kiwi-service at KIWI_SERVICE_URL (default http://localhost:8765)
 *
 * Brings two parallel workspaces up: one ingested with hybrid OFF (dense
 * baseline), one with hybrid ON. Runs the fixture queries against each and
 * asserts hybrid_hit_rate >= baseline_hit_rate.
 *
 * Boot with:
 *   ./scripts/integration/start-hybrid-stack.sh up
 *   RUN_E2E=1 \
 *     QDRANT_ENDPOINT=http://localhost:6333 \
 *     KIWI_SERVICE_URL=http://localhost:8765 \
 *     QDRANT_HYBRID_ENABLED=true \
 *     yarn test:integration
 *   ./scripts/integration/start-hybrid-stack.sh down
 */
process.env.STORAGE_DIR = process.env.STORAGE_DIR || "/tmp/anythingllm-test";
process.env.VECTOR_DB = "qdrant";

const fs = require("fs");
const path = require("path");
const http = require("http");

const SAMPLE = path.resolve(
  __dirname,
  "../../../samples/qdrant-hybrid-kiwi/raw/2026_hf_recruitment.pdf"
);
const FIXTURES = path.resolve(
  __dirname,
  "../../../samples/qdrant-hybrid-kiwi/fixtures/expected_queries.json"
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

const skip = (() => {
  if (process.env.RUN_E2E !== "1") return "RUN_E2E != 1";
  if (!fs.existsSync(SAMPLE)) return `sample PDF missing at ${SAMPLE}`;
  if (!fs.existsSync(FIXTURES)) return `fixtures missing at ${FIXTURES}`;
  return null;
})();

(skip ? describe.skip : describe)("Qdrant hybrid e2e", () => {
  jest.setTimeout(180000);

  let QDrant;
  let pdfParse;
  let helpers;

  beforeAll(async () => {
    // Last-mile reachability check: even if RUN_E2E=1, skip individual tests if
    // services aren't actually up.
    const qOk = await reachable(`${QDRANT_ENDPOINT}/readyz`);
    const kOk = await reachable(`${KIWI_URL}/healthz`);
    if (!qOk || !kOk) {
      throw new Error(
        `services not reachable: qdrant=${qOk} kiwi=${kOk} ` +
        `(QDRANT_ENDPOINT=${QDRANT_ENDPOINT}, KIWI_SERVICE_URL=${KIWI_URL})`
      );
    }
    helpers = require("../../utils/helpers");
    ({ QDrant } = require("../../utils/vectorDbProviders/qdrant"));
    pdfParse = require("pdf-parse");
  });

  async function ingest(namespace, hybrid) {
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

  async function hitRate(namespace) {
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

  const nsBase = "e2e-baseline-" + Date.now();
  const nsHyb = "e2e-hybrid-" + Date.now();

  it("hybrid hit-rate >= baseline hit-rate on the Korean PDF fixture", async () => {
    await ingest(nsBase, false);
    await ingest(nsHyb, true);
    const base = await hitRate(nsBase);
    const hyb = await hitRate(nsHyb);
    // eslint-disable-next-line no-console
    console.log(`e2e:hit-rates baseline=${base.toFixed(3)} hybrid=${hyb.toFixed(3)}`);
    expect(hyb).toBeGreaterThanOrEqual(base);
  });
});
