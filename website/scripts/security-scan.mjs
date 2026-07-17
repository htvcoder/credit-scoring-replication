import fs from "fs";
import path from "path";
import { execFileSync } from "child_process";

const repoRoot = path.resolve(process.cwd(), "..");
const gitSafeRepoRoot = repoRoot.replaceAll("\\", "/");
const ignoredDirs = new Set([
  ".git",
  ".next",
  "node_modules",
  "out",
  "__pycache__",
  ".pytest_cache",
  ".mypy_cache",
  ".ruff_cache",
]);

const forbiddenTrackedPathPatterns = [
  /^data[\\/](raw|processed)[\\/].+/i,
  /^results[\\/].+/i,
  /(^|[\\/])\.env(\.|$)/i,
  /(^|[\\/])kaggle\.json$/i,
  /api[_-]?key/i,
  /credential/i,
  /client_secret/i,
  /service-account.*\.json$/i,
  /(^|[\\/])id_(rsa|ed25519)$/i,
  /\.(pem|key)$/i,
];

const forbiddenContentPatterns = [
  { name: "windows absolute path", pattern: /[A-Z]:\\/ },
  { name: "raw data path", pattern: /data[\\/](raw|processed)/i },
  { name: "secret token marker", pattern: /(api[_-]?key|client_secret|private key|BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY)/i },
  { name: "fake AUC metric", pattern: /AUC\s*[:=]\s*0\.\d+/i },
  { name: "fake Brier metric", pattern: /Brier\s*[:=]\s*0\.\d+/i },
];

const scannedContentRoots = [
  path.join(repoRoot, "website", "app"),
  path.join(repoRoot, "website", "components"),
  path.join(repoRoot, "website", "content"),
  path.join(repoRoot, "website", "lib"),
];

function walk(dir) {
  if (!fs.existsSync(dir)) {
    return [];
  }

  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (ignoredDirs.has(entry.name)) {
        return [];
      }
      return walk(fullPath);
    }
    return fullPath;
  });
}

function fail(message) {
  console.error(message);
  process.exitCode = 1;
}

function gitRepositoryFiles() {
  try {
    const output = execFileSync(
      "git",
      [
        "-c",
        `safe.directory=${gitSafeRepoRoot}`,
        "-C",
        repoRoot,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
      ],
      {
      encoding: "utf8",
      },
    );
    return output
      .split("\0")
      .filter(Boolean)
      .map((relativePath) => path.join(repoRoot, relativePath));
  } catch {
    return walk(repoRoot);
  }
}

const repoFiles = gitRepositoryFiles();
for (const filePath of repoFiles) {
  const relativePath = path.relative(repoRoot, filePath).replaceAll("\\", "/");
  for (const pattern of forbiddenTrackedPathPatterns) {
    if (pattern.test(relativePath) && !relativePath.endsWith(".gitkeep")) {
      fail(`Forbidden repository artifact found: ${relativePath}`);
    }
  }
}

const contentFiles = scannedContentRoots
  .flatMap(walk)
  .filter((filePath) => /\.(md|json|ya?ml|tsx?|css|mjs)$/.test(filePath));

for (const filePath of contentFiles) {
  const relativePath = path.relative(repoRoot, filePath).replaceAll("\\", "/");
  const text = fs.readFileSync(filePath, "utf8");

  for (const check of forbiddenContentPatterns) {
    if (check.pattern.test(text)) {
      fail(`Forbidden ${check.name} found in ${relativePath}`);
    }
  }
}

if (!process.exitCode) {
  console.log("security scan passed");
}
