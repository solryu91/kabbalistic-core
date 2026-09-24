import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: handler } = await import(workerUrl.href);
  return handler(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
  );
}

test("server-renders the SEED product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>SEED — Local Kabbalistic Cognition Prototype<\/title>/i);
  assert.match(html, /Bring one real question into the field/);
  assert.match(html, /Begin full Tree cycle/);
  assert.match(html, /Form, Flow, and Accord/);
  assert.match(html, /Memory off/);
  assert.match(html, /not a claim of consciousness/i);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site|react-loading-skeleton/i);
});

test("renders the primary controls with semantic labels", async () => {
  const html = await (await render()).text();
  assert.match(html, /<textarea[^>]*id="intention"/i);
  assert.match(html, /<label[^>]*for="intention"/i);
  assert.match(html, /role="tablist"/i);
  assert.match(html, /aria-pressed="true"/i);
  assert.match(html, /Sources read-only/);
  assert.match(html, /No durable memory/);
  assert.match(html, /aria-describedby="symbol-hint"/i);
  assert.equal((html.match(/Project-owned functional kernel/g) ?? []).length, 3);
  assert.doesNotMatch(html, /private design reference/i);
});

test("server-renders an honest checking state and stable tab panels", async () => {
  const html = await (await render()).text();
  assert.match(html, /Checking local services/i);
  assert.match(html, /Checking sources/i);
  for (const panel of ["response", "sources", "process", "control"]) {
    assert.match(html, new RegExp(`id="panel-${panel}"`, "i"));
    assert.match(html, new RegExp(`aria-controls="panel-${panel}"`, "i"));
  }
  assert.doesNotMatch(html, /Local services need attention/);
  assert.doesNotMatch(html, />3 curated sources</);
});

test("source contracts keep local actions protected and evidence data-bound", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(page, /"X-SEED-CSRF": csrfToken/);
  assert.match(page, /new AbortController\(\)/);
  assert.match(page, /Stop waiting/);
  assert.match(page, /ArrowRight/);
  assert.match(page, /ArrowLeft/);
  assert.match(page, /event\.key === "Home"/);
  assert.match(page, /event\.key === "End"/);
  assert.match(page, /result\?\.inner_process\.route\.map/);
  assert.match(page, /boundaries\.external_actions_taken === 0/);
  assert.match(page, /boundaries\.protocol_invoked/);
  assert.match(page, /boundaries\.durable_memory_writes === 0/);
  assert.match(page, /frame\.new_relations\.join/);
  assert.match(page, /frame\.path_ids\.length > 0/);
  assert.doesNotMatch(page, /running && index === 0/);
  assert.doesNotMatch(css, /font-size:\s*(?:[6-9]|10)px/);
  assert.match(css, /\.mobile-cell-label/);
});
