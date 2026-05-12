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
    expect(hashToken("주택")).not.toBe(hashToken("금융"));
  });
});

describe("bm25.buildDocSparse", () => {
  it("produces (indices,values) with BM25 doc-side TF saturation and length norm", () => {
    const tokens = ["주택", "주택", "금융", "공사"];
    const out = buildDocSparse(tokens, { avgdl: 4, k1: 1.2, b: 0.75 });
    expect(out.indices.length).toBe(3);
    expect(out.values.length).toBe(out.indices.length);
    const idxJ = out.indices.indexOf(hashToken("주택"));
    const idxK = out.indices.indexOf(hashToken("금융"));
    expect(out.values[idxJ]).toBeGreaterThan(out.values[idxK]);
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

  it("returns a positive value when token has no df", () => {
    expect(computeIdf({ N: 100, df: 0 })).toBeCloseTo(Math.log(100 / 0.5 + 1), 6);
  });
});

describe("bm25.buildQuerySparse", () => {
  it("weights each query token by its IDF using the stats", () => {
    const tokens = ["주택", "금융"];
    const stats = {
      N: 100,
      df: { [hashToken("주택")]: 10, [hashToken("금융")]: 50 },
    };
    const out = buildQuerySparse(tokens, stats);
    expect(out.indices).toEqual(
      expect.arrayContaining([hashToken("주택"), hashToken("금융")])
    );
    const idxJ = out.indices.indexOf(hashToken("주택"));
    const idxK = out.indices.indexOf(hashToken("금융"));
    expect(out.values[idxJ]).toBeGreaterThan(out.values[idxK]);
  });
});
