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
