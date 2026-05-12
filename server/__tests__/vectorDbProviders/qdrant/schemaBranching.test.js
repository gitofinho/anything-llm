const path = require("path");

// utils/files/index.js reads STORAGE_DIR at module load time outside development env.
// Set it before any require so the path.resolve call succeeds.
process.env.STORAGE_DIR = process.env.STORAGE_DIR || "/tmp/anythingllm-test";

function loadProvider() {
  jest.resetModules();
  return require(path.join(
    "..",
    "..",
    "..",
    "utils",
    "vectorDbProviders",
    "qdrant"
  ));
}

describe("QDrant.vectorSchema", () => {
  afterEach(() => {
    delete process.env.QDRANT_HYBRID_ENABLED;
  });

  it("detects legacy unnamed-dense schema as 'dense'", async () => {
    const { QDrant } = loadProvider();
    const client = {
      getCollection: jest.fn(async () => ({
        config: { params: { vectors: { size: 1536, distance: "Cosine" } } },
      })),
    };
    const schema = await QDrant.vectorSchema(client, "legacy");
    expect(schema).toBe("dense");
  });

  it("detects named hybrid schema as 'hybrid'", async () => {
    const { QDrant } = loadProvider();
    const client = {
      getCollection: jest.fn(async () => ({
        config: {
          params: {
            vectors: { dense: { size: 1536, distance: "Cosine" } },
            sparse_vectors: { sparse: {} },
          },
        },
      })),
    };
    const schema = await QDrant.vectorSchema(client, "new");
    expect(schema).toBe("hybrid");
  });

  it("returns null when collection does not exist", async () => {
    const { QDrant } = loadProvider();
    const client = {
      getCollection: jest.fn().mockRejectedValue(new Error("not found")),
    };
    const schema = await QDrant.vectorSchema(client, "missing");
    expect(schema).toBeNull();
  });
});

describe("QDrant.getOrCreateCollection", () => {
  afterEach(() => {
    delete process.env.QDRANT_HYBRID_ENABLED;
  });

  it("creates a legacy unnamed-dense collection when hybrid disabled", async () => {
    const { QDrant } = loadProvider();
    const client = {
      getCollection: jest
        .fn()
        .mockRejectedValueOnce(new Error("not found"))
        .mockResolvedValue({}),
      createCollection: jest.fn(async () => ({ ok: true })),
      count: jest.fn(async () => ({ count: 0 })),
    };
    await QDrant.getOrCreateCollection(client, "ns", 1536);
    expect(client.createCollection).toHaveBeenCalledWith("ns", {
      vectors: { size: 1536, distance: "Cosine" },
    });
  });

  it("creates a named hybrid collection when QDRANT_HYBRID_ENABLED=true and kiwi healthy", async () => {
    process.env.QDRANT_HYBRID_ENABLED = "true";
    const { QDrant } = loadProvider();
    QDrant.__setKiwiClientForTest({ isHealthy: async () => true });
    const client = {
      getCollection: jest
        .fn()
        .mockRejectedValueOnce(new Error("not found"))
        .mockResolvedValue({}),
      createCollection: jest.fn(async () => ({ ok: true })),
      count: jest.fn(async () => ({ count: 0 })),
    };
    await QDrant.getOrCreateCollection(client, "ns", 1536);
    const [, body] = client.createCollection.mock.calls[0];
    expect(body.vectors).toEqual({ dense: { size: 1536, distance: "Cosine" } });
    expect(body.sparse_vectors).toEqual({ sparse: {} });
  });

  it("falls back to legacy when kiwi-service is unhealthy", async () => {
    process.env.QDRANT_HYBRID_ENABLED = "true";
    const { QDrant } = loadProvider();
    QDrant.__setKiwiClientForTest({ isHealthy: async () => false });
    const client = {
      getCollection: jest
        .fn()
        .mockRejectedValueOnce(new Error("not found"))
        .mockResolvedValue({}),
      createCollection: jest.fn(async () => ({ ok: true })),
      count: jest.fn(async () => ({ count: 0 })),
    };
    await QDrant.getOrCreateCollection(client, "ns", 1536);
    const [, body] = client.createCollection.mock.calls[0];
    expect(body.vectors).toEqual({ size: 1536, distance: "Cosine" });
    expect(body.sparse_vectors).toBeUndefined();
  });
});
