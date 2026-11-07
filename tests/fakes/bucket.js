// A bucket for lab/site/assets/edited-models.js: `node bucket.js <edited-models.js>` reads
// {object name: body} from stdin, serves listings and objects the way GCS does (one page,
// generation "7") and prints the loader's view of every suspect as JSON.
const em = require(process.argv[2]);
const objects = JSON.parse(require("fs").readFileSync(0, "utf8"));
const API = "https://storage.googleapis.com/storage/v1/b/" + em.BUCKET + "/o";
const GEN = "7";

function fakeFetch(url) {
  const u = new URL(url);
  let body;
  if (u.origin + u.pathname === API) {
    const prefix = u.searchParams.get("prefix"), delim = u.searchParams.get("delimiter");
    const items = [], prefixes = new Set();
    for (const name of Object.keys(objects)) {
      if (!name.startsWith(prefix)) continue;
      const rest = name.slice(prefix.length);
      if (delim && rest.includes(delim)) prefixes.add(prefix + rest.split(delim)[0] + delim);
      else items.push({ name, size: "1", updated: "2026-11-04T00:00:00Z", generation: GEN });
    }
    body = { items, prefixes: [...prefixes] };
  } else {
    const name = decodeURIComponent(u.pathname.slice(em.BUCKET.length + 2));
    if (!(name in objects) || u.searchParams.get("generation") !== GEN) return Promise.resolve({ ok: false, status: 404 });
    body = objects[name];
  }
  return Promise.resolve({ ok: true, status: 200, json: async () => body,
                           text: async () => (typeof body === "string" ? body : JSON.stringify(body)) });
}

em.load(fakeFetch).then((model) => {
  const suspects = model.suspects.map((s) => ({ sid: s.sid, error: s.error || null, status: s.error ? null : em.statusOf(s),
                                                 tiers: s.error ? null : em.scored(s),
                                                 baseContext: s.error ? null : em.baseContext(s),
                                                 blinding: s.error ? null : em.blindingHtml(s) }));
  console.log(JSON.stringify({ summary: model.summary, summaryText: em.summaryText(model.summary), suspects,
                               objectUrl: em.objectUrl("edits/s-a/weights/delta.safetensors") }));
}).catch((err) => { console.error(err); process.exit(1); });
