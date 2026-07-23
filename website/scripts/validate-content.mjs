import fs from "fs";
import path from "path";
import { parse } from "yaml";

const websiteRoot = process.cwd();
const repoRoot = path.resolve(websiteRoot, "..");
const forbiddenPatterns = [
  /data[\\/](raw|processed)/i,
  /[A-Z]:\\/,
  /(^|[\\/])\.env(\.|$)/i,
  /api[_-]?key/i,
  /credential/i,
  /client_secret/i,
  /AUC\s*[:=]\s*0\.\d+/i,
  /Brier\s*[:=]\s*0\.\d+/i,
  /XGBoost\s+(đứng đầu|tốt nhất|thắng)/i,
];

function read(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return walk(fullPath);
    }

    return fullPath;
  });
}

function fail(message) {
  console.error(message);
  process.exitCode = 1;
}

const progress = parse(read(path.join(websiteRoot, "content", "progress.yaml")));
const paper = parse(read(path.join(websiteRoot, "content", "paper.yaml")));
const publicDatasets = JSON.parse(read(path.join(websiteRoot, "content", "datasets.public.json")));
const phase0 = progress.phases.find((phase) => phase.id === "Phase 0");
const phase1 = progress.phases.find((phase) => phase.id === "Phase 1");
const phase2 = progress.phases.find((phase) => phase.id === "Phase 2");
const phase3 = progress.phases.find((phase) => phase.id === "Phase 3");
const phase4 = progress.phases.find((phase) => phase.id === "Phase 4");
const allowedStatuses = new Set(["planned", "next", "in_progress", "completed", "blocked", "deferred"]);
const expectedPhaseOrder = Array.from({ length: 12 }, (_, index) => `Phase ${index}`);

if (phase0?.status !== "completed") {
  fail("Phase 0 must be Completed.");
}

if (phase1?.status !== "completed") {
  fail("Phase 1 must be Completed after production rollback verification passed.");
}

if (phase2?.status !== "completed") {
  fail("Phase 2 must be Completed after P2A/P2B/P2C acceptance passed.");
}

if (phase3?.status !== "completed") {
  fail("Phase 3 must be marked Completed after P3A/P3B/P3C acceptance passed.");
}

if (phase3?.tag !== "p3-leakage-safe-preprocessing-complete") {
  fail("Phase 3 tag must be p3-leakage-safe-preprocessing-complete.");
}

if (phase4?.status !== "next") {
  fail("Phase 4 must be marked Next after Phase 3 completion.");
}

if (progress.phases.length !== expectedPhaseOrder.length) {
  fail(`Progress content must contain ${expectedPhaseOrder.length} phases.`);
}

for (const [index, expectedId] of expectedPhaseOrder.entries()) {
  if (progress.phases[index]?.id !== expectedId) {
    fail(`Progress phase order changed at index ${index}: expected ${expectedId}.`);
  }
}

for (const phase of progress.phases) {
  if (!allowedStatuses.has(phase.status)) {
    fail(`${phase.id} has invalid status: ${phase.status}`);
  }

  if (!phase.summary?.trim()) {
    fail(`${phase.id} summary must not be empty.`);
  }

  if (/^\s*Chưa triển khai\.?\s*$/i.test(phase.summary)) {
    fail(`${phase.id} must not use only "Chưa triển khai" as summary.`);
  }

  if (!Array.isArray(phase.tasks) || phase.tasks.length < 2) {
    fail(`${phase.id} tasks must contain at least 2 items.`);
  }

  if (!Array.isArray(phase.deliverables) || phase.deliverables.length < 1) {
    fail(`${phase.id} deliverables must contain at least 1 item.`);
  }
}

for (const phase of progress.phases.slice(5)) {
  if (phase.status !== "planned") {
    fail(`${phase.id} must remain Planned until implementation evidence exists.`);
  }
}

const requiredPaperFields = [
  "title",
  "authors",
  "year",
  "publication",
  "doi",
  "main_objective",
  "reported_findings",
];

for (const field of requiredPaperFields) {
  if (!paper[field] || (Array.isArray(paper[field]) && paper[field].length === 0)) {
    fail(`Missing required paper content field: ${field}`);
  }
}

const registryPath = path.join(repoRoot, "data", "datasets.yaml");
const registry = fs.existsSync(registryPath) ? parse(read(registryPath)) : null;
const requiredDatasetIds = ["ac", "gc", "hmeq", "th02", "tc", "gmc"];
for (const id of requiredDatasetIds) {
  const publicDataset = publicDatasets.find((dataset) => dataset.id === id.toUpperCase());
  if (!publicDataset) {
    fail(`Missing public dataset entry: ${id}`);
  }

  if (!publicDataset?.rows || !publicDataset?.inputCount || !publicDataset?.targetColumn) {
    fail(`Dataset ${id} is missing public shape metadata.`);
  }

  if (registry) {
    const dataset = registry.datasets?.[id];
    if (!dataset) {
      fail(`Missing dataset registry entry: ${id}`);
      continue;
    }

    if (
      publicDataset.rows !== dataset.expected.rows ||
      publicDataset.inputCount !== dataset.expected.input_count ||
      publicDataset.defaultRate !== dataset.expected.default_rate ||
      publicDataset.targetColumn !== dataset.target.column
    ) {
      fail(`Public dataset metadata is out of sync with data/datasets.yaml: ${id}`);
    }
  }
}

const scannedFiles = [
  ...walk(path.join(websiteRoot, "content")),
  ...walk(path.join(websiteRoot, "app")),
  ...walk(path.join(websiteRoot, "components")),
  ...walk(path.join(websiteRoot, "lib")),
].filter((filePath) => /\.(md|ya?ml|tsx?|css)$/.test(filePath));

const appRoutes = walk(path.join(websiteRoot, "app"))
  .filter((filePath) => path.basename(filePath) === "page.tsx")
  .map((filePath) =>
    path
      .relative(path.join(websiteRoot, "app"), path.dirname(filePath))
      .replaceAll("\\", "/"),
  );

if (appRoutes.includes("tai-lap")) {
  fail("Public route /tai-lap must not exist.");
}

for (const filePath of scannedFiles) {
  const relativePath = path.relative(websiteRoot, filePath);
  const text = read(filePath);

  for (const pattern of forbiddenPatterns) {
    if (pattern.test(text)) {
      fail(`Forbidden public content pattern ${pattern} found in ${relativePath}`);
    }
  }

  if (/href=["'{`]\/tai-lap\/?/.test(text) || />\s*Tái lập\s*</.test(text)) {
    fail(`Removed public route still referenced in ${relativePath}`);
  }
}

if (!process.exitCode) {
  console.log("content validation passed");
}
