const { QdrantClient } = require("@qdrant/js-client-rest");
const { TextSplitter } = require("../../TextSplitter");
const { SystemSettings } = require("../../../models/systemSettings");
const { storeVectorResult, cachedVectorInformation } = require("../../files");
const { v4: uuidv4 } = require("uuid");
const { toChunks, getEmbeddingEngineSelection } = require("../../helpers");
const { sourceIdentifier } = require("../../chats");
const { VectorDatabase } = require("../base");
const {
  KiwiClient,
  hybridConfig,
  hashToken,
  buildDocSparse,
  applyDocsDelta,
} = require("./hybrid");

let _kiwi = null;
function _kiwiClient() {
  if (_kiwi) return _kiwi;
  const cfg = hybridConfig();
  _kiwi = new KiwiClient({ baseUrl: cfg.kiwiServiceUrl });
  return _kiwi;
}

let _injectedClient = null;

class QDrant extends VectorDatabase {
  constructor() {
    super();
  }

  get name() {
    return "QDrant";
  }

  async connect() {
    if (_injectedClient) return { client: _injectedClient };
    if (process.env.VECTOR_DB !== "qdrant")
      throw new Error("QDrant::Invalid ENV settings");

    const client = new QdrantClient({
      url: process.env.QDRANT_ENDPOINT,
      ...(process.env.QDRANT_API_KEY
        ? { apiKey: process.env.QDRANT_API_KEY }
        : {}),
    });

    const isAlive = (await client.api("cluster")?.clusterStatus())?.ok || false;
    if (!isAlive)
      throw new Error(
        "QDrant::Invalid Heartbeat received - is the instance online?"
      );

    return { client };
  }

  async heartbeat() {
    await this.connect();
    return { heartbeat: Number(new Date()) };
  }

  async totalVectors() {
    const { client } = await this.connect();
    const { collections } = await client.getCollections();
    var totalVectors = 0;
    for (const collection of collections) {
      if (!collection || !collection.name) continue;
      totalVectors +=
        (await this.namespace(client, collection.name))?.vectorCount || 0;
    }
    return totalVectors;
  }

  async namespaceCount(_namespace = null) {
    const { client } = await this.connect();
    const namespace = await this.namespace(client, _namespace);
    return namespace?.vectorCount || 0;
  }

  async similarityResponse({
    client,
    namespace,
    queryVector,
    queryText = "",
    similarityThreshold = 0.25,
    topN = 4,
    filterIdentifiers = [],
  }) {
    const result = { contextTexts: [], sourceDocuments: [], scores: [] };

    const schema = await QDrant.vectorSchema(client, namespace);
    const isHybridColl = schema === "hybrid";

    if (!isHybridColl) {
      const responses = await client.search(namespace, {
        vector: queryVector,
        limit: topN,
        with_payload: true,
      });

      responses.forEach((response) => {
        if (response.score < similarityThreshold) return;
        if (filterIdentifiers.includes(sourceIdentifier(response?.payload))) {
          this.logger(
            "QDrant: A source was filtered from context as it's parent document is pinned."
          );
          return;
        }

        result.contextTexts.push(response?.payload?.text || "");
        result.sourceDocuments.push({
          ...(response?.payload || {}),
          id: response.id,
          score: response.score,
        });
        result.scores.push(response.score);
      });

      return result;
    }

    // Hybrid collection path — use Query API with RRF fusion.
    const cfg = hybridConfig();
    const kiwi = _kiwiClient();
    const kiwiHealthy = await kiwi.isHealthy();

    const prefetchLimit = Math.max(topN * 10, 50);
    const prefetch = [
      { using: "dense", query: queryVector, limit: prefetchLimit },
    ];

    if (kiwiHealthy && queryText && queryText.trim()) {
      const { readStats } = require("./hybrid/stats");
      const { buildQuerySparse } = require("./hybrid/bm25");
      const tokens = (await kiwi.tokenize([queryText], cfg.filterPos))[0] || [];
      if (tokens.length > 0) {
        const stats = await readStats(client, namespace);
        const sparseQuery = buildQuerySparse(tokens, stats);
        if (sparseQuery.indices.length > 0) {
          prefetch.push({
            using: "sparse",
            query: sparseQuery,
            limit: prefetchLimit,
          });
        }
      }
    } else if (!kiwiHealthy) {
      this.logger(
        "similarityResponse",
        `kiwi-service unhealthy; using dense-only prefetch on hybrid collection '${namespace}'.`
      );
    }

    const response = await client.query(namespace, {
      prefetch,
      query: { fusion: cfg.fusion },
      limit: topN,
      with_payload: true,
    });
    const hits = response?.points || [];

    hits.forEach((hit) => {
      if (typeof hit.score === "number" && hit.score < similarityThreshold) return;
      if (filterIdentifiers.includes(sourceIdentifier(hit?.payload))) return;
      if (!hit?.payload?.text) return;
      result.contextTexts.push(hit.payload.text);
      result.sourceDocuments.push({
        ...hit.payload,
        id: hit.id,
        score: hit.score,
      });
      result.scores.push(hit.score);
    });

    return result;
  }

  async namespace(client, namespace = null) {
    if (!namespace) throw new Error("No namespace value provided.");
    const collection = await client.getCollection(namespace).catch(() => null);
    if (!collection) return null;

    return {
      name: namespace,
      ...collection,
      vectorCount: (await client.count(namespace, { exact: true })).count,
    };
  }

  async hasNamespace(namespace = null) {
    if (!namespace) return false;
    const { client } = await this.connect();
    return await this.namespaceExists(client, namespace);
  }

  async namespaceExists(client, namespace = null) {
    if (!namespace) throw new Error("No namespace value provided.");
    const collection = await client.getCollection(namespace).catch((e) => {
      this.logger("namespaceExists", e.message);
      return null;
    });
    return !!collection;
  }

  async deleteVectorsInNamespace(client, namespace = null) {
    await client.deleteCollection(namespace);
    return true;
  }

  // QDrant requires a dimension aspect for collection creation
  // we pass this in from the first chunk to infer the dimensions like other
  // providers do.
  async getOrCreateCollection(client, namespace, dimensions = null) {
    return QDrant.getOrCreateCollection(client, namespace, dimensions, this);
  }

  static async getOrCreateCollection(client, namespace, dimensions = null, _instance = null) {
    const existingCollection = await client.getCollection(namespace).catch(() => null);
    if (existingCollection) {
      return existingCollection;
    }
    if (!dimensions)
      throw new Error(
        `Qdrant:getOrCreateCollection Unable to infer vector dimension from input.`
      );

    const cfg = hybridConfig();
    const wantHybrid = cfg.enabled && (await _kiwiClient().isHealthy());

    if (wantHybrid) {
      await client.createCollection(namespace, {
        vectors: { dense: { size: dimensions, distance: "Cosine" } },
        sparse_vectors: { sparse: {} },
      });
    } else {
      if (cfg.enabled) {
        if (_instance) {
          _instance.logger(
            "getOrCreateCollection",
            `kiwi-service unhealthy; creating legacy dense-only collection '${namespace}'.`
          );
        }
      }
      await client.createCollection(namespace, {
        vectors: { size: dimensions, distance: "Cosine" },
      });
    }
    return await client.getCollection(namespace);
  }

  async _buildHybridPoints(items) {
    const cfg = hybridConfig();
    const texts = items.map((it) => it.text || "");
    const tokenLists = await _kiwiClient().tokenize(texts, cfg.filterPos);

    const points = [];
    const docs = [];
    items.forEach((it, i) => {
      const tokens = tokenLists[i] || [];
      const hashes = tokens.map(hashToken);
      const sparse = buildDocSparse(tokens, {
        avgdl: Math.max(tokens.length, 1),
        k1: cfg.bm25.k1,
        b: cfg.bm25.b,
      });
      points.push({
        id: it.id,
        vector: { dense: it.denseVector, sparse },
        payload: it.payload,
      });
      docs.push({ tokens, hashes });
    });

    return { points, docs };
  }

  async addDocumentToNamespace(
    namespace,
    documentData = {},
    fullFilePath = null,
    skipCache = false
  ) {
    const { DocumentVectors } = require("../../../models/vectors");
    try {
      let vectorDimension = null;
      const { pageContent, docId, ...metadata } = documentData;
      if (!pageContent || pageContent.length == 0) return false;

      this.logger("Adding new vectorized document into namespace", namespace);
      if (!skipCache) {
        const cacheResult = await cachedVectorInformation(fullFilePath);
        if (cacheResult.exists) {
          const { client } = await this.connect();
          const { chunks } = cacheResult;
          const documentVectors = [];
          vectorDimension =
            chunks[0][0]?.vector?.length ??
            chunks[0][0]?.values?.length ??
            null;

          const collection = await this.getOrCreateCollection(
            client,
            namespace,
            vectorDimension
          );
          if (!collection)
            throw new Error("Failed to create new QDrant collection!", {
              namespace,
            });

          const schema = await QDrant.vectorSchema(client, namespace);
          if (schema === "hybrid") {
            for (const chunk of chunks) {
              const items = [];

              // Before sending to Qdrant and saving the records to our db
              // we need to assign the id of each chunk that is stored in the cached file.
              // The id property must be defined or else it will be unable to be managed by ALLM.
              chunk.forEach((c) => {
                const id = uuidv4();
                if (c?.payload?.hasOwnProperty("id")) {
                  const { id: _id, ...payload } = c.payload;
                  documentVectors.push({ docId, vectorId: id });
                  items.push({
                    id,
                    denseVector: c.vector,
                    payload,
                    text: payload?.text || "",
                  });
                } else {
                  console.error(
                    "The 'id' property is not defined in chunk.payload - it will be omitted from being inserted in QDrant collection."
                  );
                }
              });

              if (!items.length) continue;
              const { points, docs } = await this._buildHybridPoints(items);
              const additionResult = await client.upsert(namespace, {
                wait: true,
                points,
              });
              if (additionResult?.status !== "completed")
                throw new Error("Error embedding into QDrant", additionResult);
              await applyDocsDelta(client, namespace, docs, {
                denseDim: vectorDimension,
              });
            }
          } else {
            for (const chunk of chunks) {
              const submission = {
                ids: [],
                vectors: [],
                payloads: [],
              };

              // Before sending to Qdrant and saving the records to our db
              // we need to assign the id of each chunk that is stored in the cached file.
              // The id property must be defined or else it will be unable to be managed by ALLM.
              chunk.forEach((chunk) => {
                const id = uuidv4();
                if (chunk?.payload?.hasOwnProperty("id")) {
                  const { id: _id, ...payload } = chunk.payload;
                  documentVectors.push({ docId, vectorId: id });
                  submission.ids.push(id);
                  submission.vectors.push(chunk.vector);
                  submission.payloads.push(payload);
                } else {
                  console.error(
                    "The 'id' property is not defined in chunk.payload - it will be omitted from being inserted in QDrant collection."
                  );
                }
              });

              const additionResult = await client.upsert(namespace, {
                wait: true,
                batch: { ...submission },
              });
              if (additionResult?.status !== "completed")
                throw new Error("Error embedding into QDrant", additionResult);
            }
          }

          await DocumentVectors.bulkInsert(documentVectors);
          return { vectorized: true, error: null };
        }
      }

      // If we are here then we are going to embed and store a novel document.
      // We have to do this manually as opposed to using LangChains `Qdrant.fromDocuments`
      // because we then cannot atomically control our namespace to granularly find/remove documents
      // from vectordb.
      const EmbedderEngine = getEmbeddingEngineSelection();
      const textSplitter = new TextSplitter({
        chunkSize: TextSplitter.determineMaxChunkSize(
          await SystemSettings.getValueOrFallback({
            label: "text_splitter_chunk_size",
          }),
          EmbedderEngine?.embeddingMaxChunkLength
        ),
        chunkOverlap: await SystemSettings.getValueOrFallback(
          { label: "text_splitter_chunk_overlap" },
          20
        ),
        chunkHeaderMeta: TextSplitter.buildHeaderMeta(metadata),
        chunkPrefix: EmbedderEngine?.embeddingPrefix,
      });
      const textChunks = await textSplitter.splitText(pageContent);

      this.logger("Snippets created from document:", textChunks.length);
      const documentVectors = [];
      const vectors = [];
      const vectorValues = await EmbedderEngine.embedChunks(textChunks);
      const submission = {
        ids: [],
        vectors: [],
        payloads: [],
      };

      if (!!vectorValues && vectorValues.length > 0) {
        for (const [i, vector] of vectorValues.entries()) {
          if (!vectorDimension) vectorDimension = vector.length;
          const vectorRecord = {
            id: uuidv4(),
            vector: vector,
            // [DO NOT REMOVE]
            // LangChain will be unable to find your text if you embed manually and dont include the `text` key.
            // https://github.com/hwchase17/langchainjs/blob/2def486af734c0ca87285a48f1a04c057ab74bdf/langchain/src/vectorstores/pinecone.ts#L64
            payload: { ...metadata, text: textChunks[i] },
          };

          submission.ids.push(vectorRecord.id);
          submission.vectors.push(vectorRecord.vector);
          submission.payloads.push(vectorRecord.payload);

          vectors.push(vectorRecord);
          documentVectors.push({ docId, vectorId: vectorRecord.id });
        }
      } else {
        throw new Error(
          "Could not embed document chunks! This document will not be recorded."
        );
      }

      const { client } = await this.connect();
      const collection = await this.getOrCreateCollection(
        client,
        namespace,
        vectorDimension
      );
      if (!collection)
        throw new Error("Failed to create new QDrant collection!", {
          namespace,
        });

      if (vectors.length > 0) {
        const chunks = [];
        const schema = await QDrant.vectorSchema(client, namespace);

        if (schema === "hybrid") {
          this.logger(
            "Inserting hybrid (dense + sparse) chunks into QDrant collection."
          );
          for (const chunk of toChunks(vectors, 500)) {
            chunks.push(chunk);
            const items = chunk.map((v) => ({
              id: v.id,
              denseVector: v.vector,
              payload: v.payload,
              text: v.payload?.text || "",
            }));
            const { points, docs } = await this._buildHybridPoints(items);
            const additionResult = await client.upsert(namespace, {
              wait: true,
              points,
            });
            if (additionResult?.status !== "completed")
              throw new Error("Error embedding into QDrant", additionResult);
            await applyDocsDelta(client, namespace, docs, {
              denseDim: vectorDimension,
            });
          }
        } else {
          this.logger("Inserting vectorized chunks into QDrant collection.");
          for (const chunk of toChunks(vectors, 500)) {
            const batchIds = [],
              batchVectors = [],
              batchPayloads = [];
            chunks.push(chunk);
            chunk.forEach((v) => {
              batchIds.push(v.id);
              batchVectors.push(v.vector);
              batchPayloads.push(v.payload);
            });

            const additionResult = await client.upsert(namespace, {
              wait: true,
              batch: {
                ids: batchIds,
                vectors: batchVectors,
                payloads: batchPayloads,
              },
            });
            if (additionResult?.status !== "completed")
              throw new Error("Error embedding into QDrant", additionResult);
          }
        }

        await storeVectorResult(chunks, fullFilePath);
      }

      await DocumentVectors.bulkInsert(documentVectors);
      return { vectorized: true, error: null };
    } catch (e) {
      this.logger("addDocumentToNamespace", e.message);
      return { vectorized: false, error: e.message };
    }
  }

  async deleteDocumentFromNamespace(namespace, docId) {
    const { DocumentVectors } = require("../../../models/vectors");
    const { client } = await this.connect();
    if (!(await this.namespaceExists(client, namespace))) return;

    const knownDocuments = await DocumentVectors.where({ docId });
    if (knownDocuments.length === 0) return;

    const vectorIds = knownDocuments.map((doc) => doc.vectorId);
    await client.delete(namespace, {
      wait: true,
      points: vectorIds,
    });

    const indexes = knownDocuments.map((doc) => doc.id);
    await DocumentVectors.deleteIds(indexes);
    return true;
  }

  async performSimilaritySearch({
    namespace = null,
    input = "",
    LLMConnector = null,
    similarityThreshold = 0.25,
    topN = 4,
    filterIdentifiers = [],
  }) {
    if (!namespace || !input || !LLMConnector)
      throw new Error("Invalid request to performSimilaritySearch.");

    const { client } = await this.connect();
    if (!(await this.namespaceExists(client, namespace))) {
      return {
        contextTexts: [],
        sources: [],
        message: "Invalid query - no documents found for workspace!",
      };
    }

    const queryVector = await LLMConnector.embedTextInput(input);
    const { contextTexts, sourceDocuments } = await this.similarityResponse({
      client,
      namespace,
      queryVector,
      queryText: input,
      similarityThreshold,
      topN,
      filterIdentifiers,
    });

    const sources = sourceDocuments.map((metadata, i) => {
      return { ...metadata, text: contextTexts[i] };
    });
    return {
      contextTexts,
      sources: this.curateSources(sources),
      message: false,
    };
  }

  async "namespace-stats"(reqBody = {}) {
    const { namespace = null } = reqBody;
    if (!namespace) throw new Error("namespace required");
    const { client } = await this.connect();
    if (!(await this.namespaceExists(client, namespace)))
      throw new Error("Namespace by that name does not exist.");
    const stats = await this.namespace(client, namespace);
    return stats
      ? stats
      : { message: "No stats were able to be fetched from DB for namespace" };
  }

  async "delete-namespace"(reqBody = {}) {
    const { namespace = null } = reqBody;
    const { client } = await this.connect();
    if (!(await this.namespaceExists(client, namespace)))
      throw new Error("Namespace by that name does not exist.");

    const details = await this.namespace(client, namespace);
    await this.deleteVectorsInNamespace(client, namespace);
    return {
      message: `Namespace ${namespace} was deleted along with ${details?.vectorCount} vectors.`,
    };
  }

  async reset() {
    const { client } = await this.connect();
    const response = await client.getCollections();
    for (const collection of response.collections) {
      await client.deleteCollection(collection.name);
    }
    return { reset: true };
  }

  curateSources(sources = []) {
    const documents = [];
    for (const source of sources) {
      if (Object.keys(source).length > 0) {
        const metadata = source.hasOwnProperty("metadata")
          ? source.metadata
          : source;
        documents.push({
          ...metadata,
        });
      }
    }

    return documents;
  }

  static async vectorSchema(client, namespace) {
    const coll = await client.getCollection(namespace).catch(() => null);
    if (!coll) return null;
    const v = coll?.config?.params?.vectors;
    if (v && typeof v === "object" && v.dense && !("size" in v)) return "hybrid";
    return "dense";
  }

  static __setKiwiClientForTest(stub) {
    _kiwi = stub;
  }

  static __setQdrantClientForTest(stub) {
    _injectedClient = stub;
  }
}

module.exports.QDrant = QDrant;
