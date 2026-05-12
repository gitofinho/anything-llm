const nock = require("nock");
const {
  KiwiClient,
} = require("../../../utils/vectorDbProviders/qdrant/hybrid/kiwiClient");

describe("KiwiClient", () => {
  const base = "http://kiwi-service:8765";
  let client;

  beforeEach(() => {
    nock.disableNetConnect();
    client = new KiwiClient({ baseUrl: base, healthCacheMs: 1000 });
  });

  afterEach(() => {
    nock.cleanAll();
    nock.enableNetConnect();
  });

  it("tokenize returns the kiwi-service tokens", async () => {
    nock(base).post("/tokenize").reply(200, { tokens: [["주택", "금융"]] });
    const out = await client.tokenize(["한국주택금융"], ["NNG", "NNP"]);
    expect(out).toEqual([["주택", "금융"]]);
  });

  it("isHealthy caches the result for healthCacheMs", async () => {
    const scope = nock(base).get("/healthz").reply(200, { status: "ok" });
    expect(await client.isHealthy()).toBe(true);
    expect(await client.isHealthy()).toBe(true);
    expect(scope.isDone()).toBe(true);
  });

  it("isHealthy returns false and never throws on network error", async () => {
    nock(base).get("/healthz").replyWithError("ECONNREFUSED");
    expect(await client.isHealthy()).toBe(false);
  });

  it("tokenize throws when service returns non-2xx", async () => {
    nock(base).post("/tokenize").reply(500, { error: "boom" });
    await expect(client.tokenize(["x"], ["NNG"])).rejects.toThrow(/kiwi/i);
  });
});
