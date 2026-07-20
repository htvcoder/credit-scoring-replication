import fs from "fs";
import path from "path";
import { parse } from "yaml";

export type ResearchQuestion = {
  id: string;
  question: string;
};

export type ProjectContent = {
  title: string;
  short_description: string;
  repository: string;
  role_of_website: string;
  research_questions: ResearchQuestion[];
  objectives: string[];
  scope_notes: string[];
};

export type InternshipContent = {
  title: string;
  short_description: string;
  project_goal: string;
  work_scope: string[];
  student: string;
  supervisor: string;
  unit_or_program: string;
  timeframe: string;
  repository: string;
  website_role: string;
};

export type PaperContent = {
  title: string;
  authors: string;
  year: number;
  publication: string;
  publisher: string;
  doi: string;
  official_link: string;
  main_objective: string;
  problem_context: string;
  compared_models: string[];
  dataset_count: number;
  preprocessing: string[];
  cross_validation: string[];
  metrics: string[];
  statistical_analysis: string[];
  reported_findings: string[];
};

export type PhaseStatus = "Completed" | "In progress" | "Next" | "Planned";

export type Phase = {
  id: string;
  title: string;
  status: PhaseStatus;
  summary: string;
  tasks: string[];
  deliverables: string[];
  checkpoints?: string[];
  caveat?: string;
};

export type ProgressContent = {
  phases: Phase[];
};

const contentDir = path.join(process.cwd(), "content");

function readYaml<T>(fileName: string): T {
  const source = fs.readFileSync(path.join(contentDir, fileName), "utf8");
  return parse(source) as T;
}

export function getProjectContent() {
  return readYaml<ProjectContent>("project.yaml");
}

export function getInternshipContent() {
  return readYaml<InternshipContent>("internship.yaml");
}

export function getPaperContent() {
  return readYaml<PaperContent>("paper.yaml");
}

export function getProgressContent() {
  return readYaml<ProgressContent>("progress.yaml");
}

export function getMarkdownContent(fileName: string) {
  return fs.readFileSync(path.join(contentDir, fileName), "utf8");
}
