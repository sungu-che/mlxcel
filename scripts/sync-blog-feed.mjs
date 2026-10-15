// Copies the Backend.AI blog feeds (ko and en) into data/blog-feed-<lang>.json
// so the landing page can read them from the same origin (the feeds have no CORS header).
//
// Usage: node scripts/sync-blog-feed.mjs
// Requires Node 18+ (global fetch). No dependencies.

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const LANGS = ['ko', 'en'];
const MAX_ITEMS = 12;

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, 'data');

function feedUrl(lang) {
  return `https://www.backend.ai/${lang}/rss/blog-feed.json`;
}

async function syncLang(lang) {
  const url = feedUrl(lang);
  const res = await fetch(url, { headers: { 'User-Agent': 'mlxcel-blog-sync' } });
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status} ${res.statusText}`);
  const feed = await res.json();

  if (!Array.isArray(feed.items) || feed.items.length === 0) {
    throw new Error(`${url} has no items; keeping the previous data/blog-feed-${lang}.json.`);
  }

  // Keep only what the landing page renders; drop content_html to keep the file small.
  const items = feed.items
    .map(({ id, title, summary, url: link, image, date_modified }) => ({
      id,
      title,
      summary,
      url: link,
      image,
      date_modified,
    }))
    .sort((a, b) => new Date(b.date_modified) - new Date(a.date_modified))
    .slice(0, MAX_ITEMS);

  const out = {
    source: {
      lang,
      title: feed.title ?? null,
      feed_url: url,
      home_page_url: feed.home_page_url ?? null,
    },
    syncedAt: new Date().toISOString(),
    items,
  };
  await writeFile(path.join(outDir, `blog-feed-${lang}.json`), `${JSON.stringify(out, null, 2)}\n`, 'utf8');
  console.log(`Synced ${items.length} ${lang} blog items from ${url}`);
}

async function main() {
  await mkdir(outDir, { recursive: true });
  for (const lang of LANGS) {
    await syncLang(lang);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
