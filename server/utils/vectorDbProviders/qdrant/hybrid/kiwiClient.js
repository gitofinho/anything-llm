const _nodeFetch = require("node-fetch");

class KiwiClient {
  constructor({
    baseUrl,
    healthCacheMs = 5000,
    fetchImpl = _nodeFetch,
  } = {}) {
    if (!baseUrl) throw new Error("KiwiClient: baseUrl required");
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.healthCacheMs = healthCacheMs;
    this.fetchImpl = fetchImpl;
    this._healthCached = null;
    this._healthExpires = 0;
  }

  async tokenize(texts, filterPos = ["NNG", "NNP", "SL", "SH", "SN"]) {
    const res = await this.fetchImpl(`${this.baseUrl}/tokenize`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ texts, filterPos }),
    });
    if (!res.ok) throw new Error(`kiwi /tokenize failed: ${res.status}`);
    const body = await res.json();
    return body.tokens;
  }

  async isHealthy() {
    const now = Date.now();
    if (this._healthCached !== null && now < this._healthExpires)
      return this._healthCached;
    let ok = false;
    try {
      const res = await this.fetchImpl(`${this.baseUrl}/healthz`);
      ok = res.ok;
    } catch (_e) {
      ok = false;
    }
    this._healthCached = ok;
    this._healthExpires = now + this.healthCacheMs;
    return ok;
  }
}

module.exports = { KiwiClient };
