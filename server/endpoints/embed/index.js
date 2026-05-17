const { v4: uuidv4 } = require("uuid");
const { reqBody, multiUserMode } = require("../../utils/http");
const { Telemetry } = require("../../models/telemetry");
const { streamChatWithForEmbed } = require("../../utils/chats/embed");
const { EmbedChats } = require("../../models/embedChats");
const {
  validEmbedConfig,
  canRespond,
  setConnectionMeta,
} = require("../../utils/middleware/embedMiddleware");
const {
  convertToChatHistory,
  writeResponseChunk,
} = require("../../utils/helpers/chat/responses");
const prisma = require("../../utils/prisma");
const { fileData } = require("../../utils/files");

function embeddedEndpoints(app) {
  if (!app) return;

  app.post(
    "/embed/:embedId/stream-chat",
    [validEmbedConfig, setConnectionMeta, canRespond],
    async (request, response) => {
      try {
        const embed = response.locals.embedConfig;
        const {
          sessionId,
          message,
          // optional keys for override of defaults if enabled.
          prompt = null,
          model = null,
          temperature = null,
          username = null,
        } = reqBody(request);

        response.setHeader("Cache-Control", "no-cache");
        response.setHeader("Content-Type", "text/event-stream");
        response.setHeader("Access-Control-Allow-Origin", "*");
        response.setHeader("Connection", "keep-alive");
        response.flushHeaders();

        await streamChatWithForEmbed(response, embed, message, sessionId, {
          promptOverride: prompt,
          modelOverride: model,
          temperatureOverride: temperature,
          username,
        });
        await Telemetry.sendTelemetry("embed_sent_chat", {
          multiUserMode: multiUserMode(response),
          LLMSelection: process.env.LLM_PROVIDER || "openai",
          Embedder: process.env.EMBEDDING_ENGINE || "inherit",
          VectorDbSelection: process.env.VECTOR_DB || "lancedb",
        });
        response.end();
      } catch (e) {
        console.error(e);
        writeResponseChunk(response, {
          id: uuidv4(),
          type: "abort",
          sources: [],
          textResponse: null,
          close: true,
          error: e.message,
        });
        response.end();
      }
    }
  );

  // Embed-scoped document preview. Returns parsed text (not the original binary, which the
  // collector deletes after parsing). The widget uses this for citation chips on uploaded files.
  // Must be registered BEFORE the generic /:sessionId GET so Express matches "document" first.
  app.get(
    "/embed/:embedId/document/:docId",
    [validEmbedConfig],
    async (request, response) => {
      try {
        const embed = response.locals.embedConfig;
        const { docId } = request.params;
        const doc = await prisma.workspace_documents.findFirst({
          where: { docId, workspaceId: embed.workspace?.id },
          select: { docpath: true, filename: true, metadata: true },
        });
        if (!doc) {
          response.status(404).type("text/plain").send("Document not found");
          return;
        }
        const parsed = await fileData(doc.docpath);
        if (!parsed) {
          response.status(404).type("text/plain").send("File missing");
          return;
        }
        const meta = (() => {
          try {
            return JSON.parse(doc.metadata || "{}");
          } catch {
            return {};
          }
        })();
        const title = meta.title || doc.filename || "Document";
        const escape = (s) =>
          String(s ?? "").replace(
            /[&<>"']/g,
            (c) =>
              ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;",
              })[c]
          );
        response.set("Cache-Control", "no-store");
        response.type("text/html").send(`<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>${escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;line-height:1.6}h1{font-size:18px;border-bottom:1px solid #eee;padding-bottom:8px}.meta{color:#666;font-size:12px;margin-bottom:16px}pre{white-space:pre-wrap;word-wrap:break-word;background:#fafafa;padding:16px;border-radius:8px;font-family:inherit}</style>
</head><body>
<h1>${escape(title)}</h1>
<div class="meta">${escape(meta.docSource || "")}${meta.published ? " · " + escape(meta.published) : ""}</div>
<pre>${escape(parsed.pageContent || "")}</pre>
</body></html>`);
      } catch (e) {
        console.error("[embed/document]", e);
        response.status(500).type("text/plain").send("error");
      }
    }
  );

  app.get(
    "/embed/:embedId/:sessionId",
    [validEmbedConfig],
    async (request, response) => {
      try {
        const { sessionId } = request.params;
        const embed = response.locals.embedConfig;
        const history = await EmbedChats.forEmbedByUser(
          embed.id,
          sessionId,
          null,
          null,
          true
        );

        response.status(200).json({ history: convertToChatHistory(history) });
      } catch (e) {
        console.error(e.message, e);
        response.sendStatus(500).end();
      }
    }
  );

  app.delete(
    "/embed/:embedId/:sessionId",
    [validEmbedConfig],
    async (request, response) => {
      try {
        const { sessionId } = request.params;
        const embed = response.locals.embedConfig;

        await EmbedChats.markHistoryInvalid(embed.id, sessionId);
        response.status(200).end();
      } catch (e) {
        console.error(e.message, e);
        response.sendStatus(500).end();
      }
    }
  );
}

module.exports = { embeddedEndpoints };
