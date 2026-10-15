// Vercel serverless function: proxies the Backend.AI blog feed with a CORS header,
// so the landing page on GitHub Pages can read it directly.
//
// GET /api/blog-feed?lang=ko|en
// Response: { source, syncedAt, items } (same shape as data/blog-feed-<lang>.json)

const LANGS = ['ko', 'en'];
const MAX_ITEMS = 12;

function feedUrl(lang) {
  return `https://www.backend.ai/${lang}/rss/blog-feed.json`;
}

module.exports = async function handler(req, res) {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.setHeader('Allow', 'GET, HEAD');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const lang = String((req.query && req.query.lang) || 'ko');
  if (!LANGS.includes(lang)) {
    return res.status(400).json({ error: 'lang must be ko or en' });
  }

  const url = feedUrl(lang);
  try {
    const upstream = await fetch(url, { headers: { 'User-Agent': 'mlxcel-blog-proxy' } });
    if (!upstream.ok) {
      return res.status(502).json({ error: `Upstream ${upstream.status}` });
    }
    const feed = await upstream.json();
    if (!Array.isArray(feed.items) || feed.items.length === 0) {
      return res.status(502).json({ error: 'Upstream feed has no items' });
    }

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

    // Cache at the edge for 30 minutes; serve stale for a day while revalidating.
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Cache-Control', 's-maxage=1800, stale-while-revalidate=86400');
    return res.status(200).json({
      source: { lang, title: feed.title ?? null, feed_url: url, home_page_url: feed.home_page_url ?? null },
      syncedAt: new Date().toISOString(),
      items,
    });
  } catch (err) {
    return res.status(502).json({ error: 'Failed to load upstream feed' });
  }
};
