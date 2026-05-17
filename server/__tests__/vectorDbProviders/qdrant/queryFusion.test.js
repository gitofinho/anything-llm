process.env.STORAGE_DIR = "/tmp/anythingllm-test";
process.env.VECTOR_DB = "qdrant";

jest.mock("../../../utils/files", () => ({
  cachedVectorInformation: async () => ({ exists: false }),
  storeVectorResult: async () => null,
}));

jest.mock("../../../utils/chats", () => ({
  sourceIdentifier: (p) => p?.docId || null,
}));

const { QDrant } = require("../../../utils/vectorDbProviders/qdrant");

function fakeHits(count) {
  return Array.from({ length: count }).map((_, i) => ({
    id: `c-${i}`,
    score: 0.9 - i * 0.05,
    payload: { text: `chunk ${i}`, title: "t" },
  }));
}

describe("QDrant.similarityResponse — hybrid path", () => {
  let db;
  beforeEach(() => {
    process.env.QDRANT_HYBRID_ENABLED = "true";
    QDrant.__setKiwiClientForTest({
      isHealthy: async () => true,
      tokenize: async () => [["주택", "금융"]],
    });
    db = new QDrant();
  });
  afterEach(() => {
    QDrant.__setKiwiClientForTest(null);
    delete process.env.QDRANT_HYBRID_ENABLED;
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
      retrieve: jest.fn(async () => [
        { payload: { N: 2, totalLen: 4, df: {} } },
      ]),
      query: jest.fn(async () => ({ points: fakeHits(3) })),
      search: jest.fn(),
    };

    const res = await db.similarityResponse({
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

  it("propagates the qdrant point id onto each sourceDocument (matches dense-path contract for docId resolution)", async () => {
    const client = {
      getCollection: jest.fn(async () => ({
        config: {
          params: {
            vectors: { dense: { size: 4, distance: "Cosine" } },
            sparse_vectors: { sparse: {} },
          },
        },
      })),
      retrieve: jest.fn(async () => [
        { payload: { N: 2, totalLen: 4, df: {} } },
      ]),
      query: jest.fn(async () => ({ points: fakeHits(3) })),
      search: jest.fn(),
    };

    const res = await db.similarityResponse({
      client,
      namespace: "ns",
      queryVector: [0.1, 0.2, 0.3, 0.4],
      queryText: "주택 금융 채용",
      topN: 3,
    });

    // The embed citation chain resolves source.id -> document_vectors.vectorId -> docId.
    // The dense path sets `id: response.id`; the hybrid path must do the same.
    expect(res.sourceDocuments.map((s) => s.id)).toEqual(["c-0", "c-1", "c-2"]);
  });

  it("falls back to dense-only client.search on legacy collections", async () => {
    const client = {
      getCollection: jest.fn(async () => ({
        config: { params: { vectors: { size: 4, distance: "Cosine" } } },
      })),
      query: jest.fn(),
      search: jest.fn(async () => fakeHits(2)),
    };

    const res = await db.similarityResponse({
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

  it("falls back to dense-only prefetch when kiwi is unhealthy on a hybrid collection", async () => {
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
      retrieve: jest.fn(async () => [
        { payload: { N: 1, totalLen: 2, df: {} } },
      ]),
      query: jest.fn(async () => ({ points: fakeHits(2) })),
      search: jest.fn(),
    };

    await db.similarityResponse({
      client,
      namespace: "ns",
      queryVector: [0.1, 0.2, 0.3, 0.4],
      queryText: "X",
      topN: 2,
    });

    const [, body] = client.query.mock.calls[0];
    expect(body.prefetch.length).toBe(1);
    expect(body.prefetch[0].using).toBe("dense");
  });

  it("uses dense-only prefetch when query text is empty even if kiwi is healthy", async () => {
    const client = {
      getCollection: jest.fn(async () => ({
        config: {
          params: {
            vectors: { dense: { size: 4, distance: "Cosine" } },
            sparse_vectors: { sparse: {} },
          },
        },
      })),
      retrieve: jest.fn(async () => [{ payload: { N: 1, totalLen: 1, df: {} } }]),
      query: jest.fn(async () => ({ points: fakeHits(2) })),
      search: jest.fn(),
    };

    await db.similarityResponse({
      client,
      namespace: "ns",
      queryVector: [0.1, 0.2, 0.3, 0.4],
      queryText: "",
      topN: 2,
    });

    const [, body] = client.query.mock.calls[0];
    expect(body.prefetch.length).toBe(1);
    expect(body.prefetch[0].using).toBe("dense");
  });
});
