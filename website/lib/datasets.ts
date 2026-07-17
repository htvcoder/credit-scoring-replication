import fs from "fs";
import path from "path";

export type DatasetSummary = {
  id: string;
  fullName: string;
  rows: number;
  inputCount: number;
  targetColumn: string;
  defaultRate: number;
  source: string;
  sourceUrl: string;
  license: string;
  usable: boolean;
  publicNote: string;
};

export function getDatasetSummaries(): DatasetSummary[] {
  const publicRegistryPath = path.join(process.cwd(), "content", "datasets.public.json");
  const source = fs.readFileSync(publicRegistryPath, "utf8");
  return JSON.parse(source) as DatasetSummary[];
}
