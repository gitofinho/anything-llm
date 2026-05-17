/**
 * Replace all documents in workspace 1 with a single PDF, re-embedding into
 * the active vector store (per server/.env.development).
 *
 * Usage:
 *   node scripts/swap-workspace-doc.js "/abs/path/to/file.pdf" [workspaceId]
 *
 * Must be run from the repository root; loads server/.env.development.
 */
const path = require("path");
const fs = require("fs");

const SERVER_DIR = path.resolve(__dirname, "../server");
require(path.join(SERVER_DIR, "node_modules/dotenv")).config({
  path: path.join(SERVER_DIR, ".env.development"),
});
process.chdir(SERVER_DIR);

const { v4: uuidv4 } = require(path.join(SERVER_DIR, "node_modules/uuid"));
const pdfParse = require(path.join(SERVER_DIR, "node_modules/pdf-parse"));
const prisma = require(path.join(SERVER_DIR, "utils/prisma"));
const { getVectorDbClass } = require(path.join(SERVER_DIR, "utils/helpers"));
const Document = require(path.join(SERVER_DIR, "models/documents")).Document;
const { DocumentVectors } = require(path.join(SERVER_DIR, "models/vectors"));

const PDF_PATH = process.argv[2];
const WORKSPACE_ID = Number(process.argv[3] || 1);
if (!PDF_PATH || !fs.existsSync(PDF_PATH)) {
  console.error("PDF not found:", PDF_PATH);
  process.exit(1);
}

const DOCUMENTS_DIR = path.join(SERVER_DIR, "storage/documents");
const CUSTOM_DIR = path.join(DOCUMENTS_DIR, "custom-documents");
if (!fs.existsSync(CUSTOM_DIR)) fs.mkdirSync(CUSTOM_DIR, { recursive: true });

function sanitizeFilenameSegment(name) {
  return name
    .normalize("NFKD")
    .replace(/[\s]+/g, "-")
    .replace(/[^\w.\-]+/g, "")
    .slice(0, 60) || "doc";
}

(async () => {
  const workspace = await prisma.workspaces.findUnique({
    where: { id: WORKSPACE_ID },
  });
  if (!workspace) throw new Error(`Workspace ${WORKSPACE_ID} not found`);
  console.log(`Workspace: id=${workspace.id} slug=${workspace.slug} name=${workspace.name}`);

  const VectorDb = getVectorDbClass();
  console.log(`Vector DB: ${process.env.VECTOR_DB} (class=${VectorDb.name || VectorDb.constructor?.name || "?"})`);

  // --- 1. Remove existing documents ---
  const existing = await Document.forWorkspace(workspace.id);
  console.log(`Removing ${existing.length} existing document(s)`);
  for (const doc of existing) {
    try {
      await VectorDb.deleteDocumentFromNamespace(workspace.slug, doc.docId);
    } catch (e) {
      console.warn(`  vector delete failed for ${doc.docId}: ${e.message}`);
    }
    try {
      await DocumentVectors.delete({ docId: doc.docId });
    } catch (e) {
      console.warn(`  vector-row delete failed for ${doc.docId}: ${e.message}`);
    }
    const fullPath = path.join(DOCUMENTS_DIR, doc.docpath);
    if (fs.existsSync(fullPath)) fs.unlinkSync(fullPath);
    await prisma.workspace_documents.delete({ where: { id: doc.id } });
    console.log(`  removed: ${doc.filename}`);
  }

  // --- 2. Parse PDF ---
  console.log(`Parsing PDF: ${PDF_PATH}`);
  const buf = fs.readFileSync(PDF_PATH);
  const parsed = await pdfParse(buf);
  const text = parsed.text.replace(/\s+/g, " ").trim();
  if (!text || text.length === 0) throw new Error("PDF parsed empty");
  console.log(`  pages=${parsed.numpages} chars=${text.length} words≈${text.split(/\s+/).length}`);

  // --- 3. Write document JSON ---
  const title = path.basename(PDF_PATH);
  const docId = uuidv4();
  const filename = `${sanitizeFilenameSegment(title)}-${docId}.json`;
  const docPath = `custom-documents/${filename}`;
  const fullDocPath = path.join(DOCUMENTS_DIR, docPath);

  const docJson = {
    id: docId,
    url: `file://${PDF_PATH}`,
    title,
    docAuthor: (parsed.info && parsed.info.Author) || "unknown",
    description: "No description found.",
    docSource: "pdf file uploaded by the user.",
    chunkSource: "",
    published: new Date().toLocaleString("en-US"),
    wordCount: text.split(/\s+/).length,
    pageContent: text,
    token_count_estimate: Math.ceil(text.length / 4),
  };
  fs.writeFileSync(fullDocPath, JSON.stringify(docJson, null, 2));
  console.log(`Wrote: ${docPath}`);

  // --- 4. Embed + register ---
  console.log("Embedding via Document.addDocuments…");
  const result = await Document.addDocuments(workspace, [docPath], null);
  console.log(`Done. embedded=${result.embedded?.length || 0} failed=${result.failedToEmbed?.length || 0}`);
  if (result.errors && result.errors.length) console.log("Errors:", result.errors);

  await prisma.$disconnect();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
