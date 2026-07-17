import fs from "fs";
import path from "path";
import { parse } from "yaml";

const websiteRoot = process.cwd();
const repoRoot = path.resolve(websiteRoot, "..");
const registryPath = path.join(repoRoot, "data", "datasets.yaml");
const outputPath = path.join(websiteRoot, "content", "datasets.public.json");
const orderedDatasetIds = ["ac", "gc", "hmeq", "th02", "tc", "gmc"];

function publicDatasetNote(dataset) {
  if (dataset.id === "hmeq") {
    return "Caveat: artifact full khớp shape/schema/class distribution của paper nhưng checksum không phải checksum SAS artifact kỳ vọng.";
  }

  if (dataset.id === "ac") {
    return "Caveat: target semantics được suy luận từ phân phối class và prior default rate của paper.";
  }

  if (dataset.id === "th02") {
    return "Caveat: raw workbook là Excel legacy; CSV conversion chỉ là format artifact, chưa áp dụng experimental preprocessing.";
  }

  if (dataset.id === "tc") {
    return "Caveat: ID không phải input mô hình.";
  }

  if (dataset.id === "gmc") {
    return "Caveat: index column không phải input mô hình; cần tuân thủ điều kiện truy cập Kaggle.";
  }

  return dataset.deviation_notes && dataset.deviation_notes !== "None."
    ? dataset.deviation_notes
    : "Không có deviation công khai trọng yếu ở Phase 0.";
}

function toPublicDataset(dataset) {
  return {
    id: dataset.id.toUpperCase(),
    fullName: dataset.full_name,
    rows: dataset.expected.rows,
    inputCount: dataset.expected.input_count,
    targetColumn: dataset.target.column,
    defaultRate: dataset.expected.default_rate,
    source: dataset.source,
    sourceUrl: dataset.source_url,
    license: dataset.license,
    usable: dataset.usable,
    publicNote: publicDatasetNote(dataset),
  };
}

const registry = parse(fs.readFileSync(registryPath, "utf8"));
const publicDatasets = orderedDatasetIds.map((id) => {
  const dataset = registry.datasets?.[id];
  if (!dataset) {
    throw new Error(`Missing dataset registry entry: ${id}`);
  }

  return toPublicDataset(dataset);
});

const serialized = `${JSON.stringify(publicDatasets, null, 2)}\n`;
fs.writeFileSync(outputPath, serialized, "utf8");
console.log(`wrote ${path.relative(websiteRoot, outputPath)}`);
