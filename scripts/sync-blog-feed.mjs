// Copies the Backend.AI blog feed into data/blog-feed.json so the landing page
// can read it from the same origin (the feed has no CORS header).
//
// Usage: node scripts/sync-blog-feed.mjs
// Requires Node 18+ (global fetch). No dependencies.

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const FEED_URL = 'https://www.backend.ai/ko/rss/blog-feed.json';
const MAX_ITEMS = 12;

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, 'data');

async function main() {
  const res = await fetch(FEED_URL, { headers: { 'User-Agent': 'mlxcel-blog-sync' } });
  if (!res.ok) throw new Error(`GET ${FEED_URL} -> ${res.status} ${res.statusText}`);
  const feed = await res.json();

  if (!Array.isArray(feed.items) || feed.items.length === 0) {
    throw new Error('Blog feed has no items; keeping the previous data/blog-feed.json.');
  }

  // Keep only what the landing page renders; drop content_html to keep the file small.
  const items = feed.items
    .map(({ id, title, summary, url, image, date_modified }) => ({
      id,
      title,
      summary,
      url,
      image,
      date_modified,
    }))
    .sort((a, b) => new Date(b.date_modified) - new Date(a.date_modified))
    .slice(0, MAX_ITEMS);

  await mkdir(outDir, { recursive: true });
  const out = {
    source: {
      title: feed.title ?? null,
      feed_url: FEED_URL,
      home_page_url: feed.home_page_url ?? null,
    },
    syncedAt: new Date().toISOString(),
    items,
  };
  await writeFile(path.join(outDir, 'blog-feed.json'), `${JSON.stringify(out, null, 2)}\n`, 'utf8');

  console.log(`Synced ${items.length} blog items from ${FEED_URL}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
