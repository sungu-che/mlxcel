// Mirrors the Markdown files under lablup/mlxcel `docs/` into `docs-content/`
// so the static docs page (docs.html) can render them.
//
// Usage: node scripts/sync-docs.mjs
// Optional env: GITHUB_TOKEN (raises the GitHub API rate limit).
// Requires Node 18+ (global fetch). No dependencies.

import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SOURCE_REPO = 'lablup/mlxcel';
const SOURCE_REF = 'main';
const SOURCE_DIR = 'docs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, 'docs-content');

const headers = {
  Accept: 'application/vnd.github+json',
  'User-Agent': 'mlxcel-docs-sync',
};
if (process.env.GITHUB_TOKEN) {
  headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
}

async function getJson(url) {
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status} ${res.statusText}`);
  return res.json();
}

async function getText(url) {
  const res = await fetch(url, { headers: { 'User-Agent': headers['User-Agent'] } });
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status} ${res.statusText}`);
  return res.text();
}

// First level-1 heading, falling back to the file name.
function extractTitle(markdown, fallback) {
  const match = markdown.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallback;
}

async function main() {
  const commit = await getJson(
    `https://api.github.com/repos/${SOURCE_REPO}/commits/${SOURCE_REF}`,
  );

  const listing = await getJson(
    `https://api.github.com/repos/${SOURCE_REPO}/contents/${SOURCE_DIR}?ref=${SOURCE_REF}`,
  );

  // Only top-level Markdown files; subdirectories (adr/, benchmark_results/, ...) are skipped.
  const mdFiles = listing.filter((entry) => entry.type === 'file' && entry.name.endsWith('.md'));

  const docs = [];
  for (const entry of mdFiles) {
    const markdown = await getText(entry.download_url);
    const slug = entry.name.replace(/\.md$/, '');
    docs.push({
      name: entry.name,
      slug,
      title: extractTitle(markdown, slug),
      size: entry.size,
      markdown,
    });
  }

  if (docs.length === 0) {
    throw new Error(`No Markdown files found in ${SOURCE_REPO}/${SOURCE_DIR}; refusing to wipe output.`);
  }

  // Replace the output directory contents so removed upstream files disappear too.
  await mkdir(outDir, { recursive: true });
  for (const existing of await readdir(outDir)) {
    if (existing.endsWith('.md') || existing === 'index.json') {
      await rm(path.join(outDir, existing));
    }
  }

  for (const doc of docs) {
    await writeFile(path.join(outDir, doc.name), doc.markdown, 'utf8');
  }

  // Sort by title for a stable, readable menu. Written last so a failed run never
  // leaves an index that points at missing files.
  const index = {
    source: {
      repo: SOURCE_REPO,
      ref: SOURCE_REF,
      dir: SOURCE_DIR,
      commit: commit.sha,
      commitDate: commit.commit?.committer?.date ?? null,
    },
    files: docs
      .map(({ name, slug, title, size }) => ({ name, slug, title, size }))
      .sort((a, b) => a.title.localeCompare(b.title)),
  };
  await writeFile(path.join(outDir, 'index.json'), `${JSON.stringify(index, null, 2)}\n`, 'utf8');

  console.log(`Synced ${docs.length} docs from ${SOURCE_REPO}@${commit.sha.slice(0, 7)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
