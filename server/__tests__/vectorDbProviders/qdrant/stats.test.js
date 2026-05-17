const {
  STATS_POINT_ID,
  readStats,
  applyDocsDelta,
} = require("../../../utils/vectorDbProviders/qdrant/hybrid/stats");

function makeClientStub(seedPayload = null) {
  return {
    retrieve: jest.fn(async (_ns, { ids }) => {
      if (!seedPayload) return [];
      return ids.includes(STATS_POINT_ID)
        ? [{ id: STATS_POINT_ID, payload: seedPayload }]
        : [];
    }),
    upsert: jest.fn(async () => ({ status: "ok" })),
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
  it("accumulates N, totalLen, and per-token df with denseDim padding", async () => {
    const client = makeClientStub({ N: 1, totalLen: 4, df: { 7: 1 } });
    const docs = [
      { tokens: ["주택", "주택", "금융"], hashes: [7, 7, 9] },
      { tokens: ["공사"], hashes: [11] },
    ];
    await applyDocsDelta(client, "ns", docs, { denseDim: 4 });
    expect(client.upsert).toHaveBeenCalledTimes(1);
    const point = client.upsert.mock.calls[0][1].points[0];
    expect(point.id).toBe(STATS_POINT_ID);
    expect(point.payload.N).toBe(3);
    expect(point.payload.totalLen).toBe(4 + 3 + 1);
    expect(point.payload.df["7"]).toBe(2);
    expect(point.payload.df["9"]).toBe(1);
    expect(point.payload.df["11"]).toBe(1);
    expect(point.vector.dense).toHaveLength(4);
    expect(point.vector.sparse).toEqual(
      expect.objectContaining({ indices: expect.any(Array), values: expect.any(Array) })
    );
  });
});
