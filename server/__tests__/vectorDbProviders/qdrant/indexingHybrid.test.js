process.env.STORAGE_DIR = "/tmp/anythingllm-test";
process.env.VECTOR_DB = "qdrant";

jest.mock("../../../utils/helpers", () => {
  const actual = jest.requireActual("../../../utils/helpers");
  return {
    ...actual,
    getEmbeddingEngineSelection: () => ({
      embedderLimit: 8,
      embeddingMaxChunkLength: 1000,
      embeddingPrefix: "",
      embedTextInput: async () => [0.1, 0.2, 0.3, 0.4],
      embedChunks: async (chunks) =>
        chunks.map(() => [0.1, 0.2, 0.3, 0.4]),
    }),
  };
});

jest.mock("../../../models/systemSettings", () => ({
  SystemSettings: {
    getValueOrFallback: async ({ label }, fb) =>
      label === "text_splitter_chunk_size" ? 200 : (fb ?? 20),
  },
}));

jest.mock("../../../models/vectors", () => ({
  DocumentVectors: { bulkInsert: jest.fn(async () => null) },
}));

jest.mock("../../../utils/files", () => ({
  cachedVectorInformation: async () => ({ exists: false }),
  storeVectorResult: async () => null,
}));

jest.mock("../../../utils/chats", () => ({
  sourceIdentifier: (p) => p?.docId || null,
}));

const { STATS_POINT_ID } = require("../../../utils/vectorDbProviders/qdrant/hybrid/stats");
const { QDrant } = require("../../../utils/vectorDbProviders/qdrant");

function hybridClientStub() {
  const upserts = [];
  return {
    upserts,
    api: () => ({ clusterStatus: async () => ({ ok: true }) }),
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
      return { status: "completed" };
    }),
    createCollection: jest.fn(async () => ({ ok: true })),
  };
}

function legacyClientStub() {
  return {
    api: () => ({ clusterStatus: async () => ({ ok: true }) }),
    getCollection: jest.fn(async () => ({
      config: { params: { vectors: { size: 4, distance: "Cosine" } } },
    })),
    count: jest.fn(async () => ({ count: 0 })),
    upsert: jest.fn(async () => ({ status: "completed" })),
    createCollection: jest.fn(async () => ({ ok: true })),
  };
}

describe("QDrant.addDocumentToNamespace — hybrid", () => {
  beforeEach(() => {
    process.env.QDRANT_HYBRID_ENABLED = "true";
    QDrant.__setKiwiClientForTest({
      isHealthy: async () => true,
      tokenize: async (texts) => texts.map(() => ["주택", "금융", "공사"]),
    });
  });
  afterEach(() => {
    QDrant.__setQdrantClientForTest(null);
    QDrant.__setKiwiClientForTest(null);
    delete process.env.QDRANT_HYBRID_ENABLED;
  });

  it("upserts points with named vectors {dense, sparse} on a hybrid collection", async () => {
    const client = hybridClientStub();
    QDrant.__setQdrantClientForTest(client);
    const db = new QDrant();

    const result = await db.addDocumentToNamespace(
      "ns",
      {
        docId: "doc-1",
        pageContent: "한국주택금융공사 채용계획 본문 ".repeat(20),
        metadata: { title: "t" },
      },
      "/tmp/cache/path"
    );

    expect(result?.vectorized).toBe(true);
    const contentUpserts = client.upsert.mock.calls.filter(
      ([_ns, body]) => body.points && body.points[0].id !== STATS_POINT_ID
    );
    expect(contentUpserts.length).toBeGreaterThan(0);
    const point = contentUpserts[0][1].points[0];
    expect(point.vector).toEqual(
      expect.objectContaining({
        dense: expect.any(Array),
        sparse: expect.objectContaining({
          indices: expect.any(Array),
          values: expect.any(Array),
        }),
      })
    );
  });

  it("updates the __bm25_stats__ reserved point with N>=1", async () => {
    const client = hybridClientStub();
    QDrant.__setQdrantClientForTest(client);
    const db = new QDrant();

    await db.addDocumentToNamespace(
      "ns",
      { docId: "doc-1", pageContent: "한국주택 금융", metadata: {} },
      "/tmp/x"
    );

    const statsUpserts = client.upsert.mock.calls.filter(
      ([_ns, body]) =>
        body.points && body.points[0].id === STATS_POINT_ID
    );
    expect(statsUpserts.length).toBeGreaterThanOrEqual(1);
    expect(statsUpserts[0][1].points[0].payload.N).toBeGreaterThanOrEqual(1);
  });

  it("falls back to legacy unnamed-dense batch upsert on a legacy collection", async () => {
    const client = legacyClientStub();
    QDrant.__setQdrantClientForTest(client);
    const db = new QDrant();

    await db.addDocumentToNamespace(
      "ns",
      { docId: "doc-1", pageContent: "X".repeat(100), metadata: {} },
      "/tmp/y"
    );

    const call = client.upsert.mock.calls[0][1];
    expect(call.batch).toBeDefined();
    expect(Array.isArray(call.batch.vectors)).toBe(true);
    expect(call.points).toBeUndefined();
  });
});
