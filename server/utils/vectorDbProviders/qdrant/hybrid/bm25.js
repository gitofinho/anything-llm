const SPARSE_DIM_BITS = 20;
const SPARSE_DIM_MASK = (1 << SPARSE_DIM_BITS) - 1;

function hashToken(token) {
  let h = 0x811c9dc5;
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h & SPARSE_DIM_MASK;
}

function buildDocSparse(tokens, { avgdl, k1, b }) {
  if (!tokens.length) return { indices: [], values: [] };
  const dl = tokens.length;
  const tf = new Map();
  for (const t of tokens) {
    const k = hashToken(t);
    tf.set(k, (tf.get(k) || 0) + 1);
  }
  const indices = [];
  const values = [];
  const norm = 1 - b + b * (dl / Math.max(avgdl, 1));
  for (const [idx, f] of tf.entries()) {
    indices.push(idx);
    values.push(((k1 + 1) * f) / (k1 * norm + f));
  }
  return { indices, values };
}

function computeIdf({ N, df }) {
  if (!df) return Math.log(N / 0.5 + 1);
  return Math.log(((N - df + 0.5) / (df + 0.5)) + 1);
}

function buildQuerySparse(tokens, stats) {
  if (!tokens.length) return { indices: [], values: [] };
  const N = Math.max(stats.N || 0, 1);
  const dfMap = stats.df || {};
  const buckets = new Map();
  for (const t of tokens) {
    const idx = hashToken(t);
    const df = dfMap[idx] || 0;
    const weight = computeIdf({ N, df });
    buckets.set(idx, (buckets.get(idx) || 0) + weight);
  }
  return {
    indices: [...buckets.keys()],
    values: [...buckets.values()],
  };
}

module.exports = {
  hashToken,
  buildDocSparse,
  buildQuerySparse,
  computeIdf,
  SPARSE_DIM_BITS,
};
