// A single reserved point per Qdrant collection holds BM25 statistics.
// id is a fixed UUID so it doesn't collide with content points
// (AnythingLLM uses random uuidv4 for content).
const STATS_POINT_ID = "00000000-0000-4000-8000-000000bm2500";

const _locks = new Map(); // namespace -> Promise chain

function _withLock(namespace, fn) {
  const prev = _locks.get(namespace) || Promise.resolve();
  const next = prev.catch(() => {}).then(fn);
  _locks.set(
    namespace,
    next.finally(() => {
      if (_locks.get(namespace) === next) _locks.delete(namespace);
    })
  );
  return next;
}

async function readStats(client, namespace) {
  const points = await client.retrieve(namespace, {
    ids: [STATS_POINT_ID],
    with_payload: true,
    with_vector: false,
  });
  if (!points || points.length === 0) return { N: 0, totalLen: 0, df: {} };
  const { N = 0, totalLen = 0, df = {} } = points[0].payload || {};
  return { N, totalLen, df };
}

async function _writeStats(client, namespace, stats, denseDim) {
  // Hybrid collections require both dense and sparse vectors. Use a zero
  // dense vector of the correct dimension so the upsert is accepted.
  const dense = denseDim ? new Array(denseDim).fill(0) : [];
  const vector = {
    dense,
    sparse: { indices: [0], values: [0] },
  };
  await client.upsert(namespace, {
    points: [
      {
        id: STATS_POINT_ID,
        payload: { __bm25_stats__: true, ...stats },
        vector,
      },
    ],
  });
}

async function applyDocsDelta(client, namespace, docs, { denseDim } = {}) {
  return _withLock(namespace, async () => {
    const stats = await readStats(client, namespace);
    for (const doc of docs) {
      stats.N += 1;
      stats.totalLen += doc.tokens.length;
      const seen = new Set();
      for (const h of doc.hashes) {
        if (seen.has(h)) continue;
        seen.add(h);
        stats.df[h] = (stats.df[h] || 0) + 1;
      }
    }
    await _writeStats(client, namespace, stats, denseDim);
    return stats;
  });
}

function avgdl(stats) {
  return stats.N > 0 ? stats.totalLen / stats.N : 1;
}

module.exports = { STATS_POINT_ID, readStats, applyDocsDelta, avgdl };
