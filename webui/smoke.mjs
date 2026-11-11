// Headless walk through every page: fails on console errors or uncaught exceptions, saves
// screenshots under webui/.shots/. Run with the API up:  node smoke.mjs [http://127.0.0.1:8601]
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://127.0.0.1:8601";
mkdirSync(".shots", { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(`console: ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("response", (r) => { if (r.url().includes("/api/") && r.status() >= 400) errors.push(`api ${r.status()} ${r.url()}`); });

const shot = (name) => page.screenshot({ path: `.shots/${name}.png`, fullPage: true });
const text = async () => (await page.locator("main").innerText()).replace(/\s+/g, " ");

await page.goto(base + "/");
await page.waitForSelector("text=Should this model run on a GPU");
await shot("01-home");

// wizard
await page.goto(base + "/new");
await page.waitForSelector("text=1 · What do you want to size hardware for?");
await page.locator(".tile:has-text('Sensitive-data detection')").first().click();
await shot("02-step1");
await page.click("text=Next →");
await page.waitForSelector("text=2 · Which test data?");
await page.waitForSelector("details.doc", { timeout: 30000 });
await shot("03-step2-data");
await page.click("text=Next →");
await page.locator(".tile:has-text('Full test')").first().click();
await page.click("text=Next →");
await page.waitForSelector("text=4 · Which models?");
await page.locator(".tile:has-text('Roblox PII classifier v2')").first().click();
await page.locator(".tile:has-text('Qwen3 4B Instruct')").first().click();
await shot("04-step4-models");
await page.click("text=Next →");
await page.waitForSelector("text=5 · Which hardware to compare?");
// tick the GPU option and the vLLM CPU option
const gpuTile = page.locator(".tile.gpu").first();
await gpuTile.locator(".t").click();
const cpuTile = page.locator(".tile.cpu").first();
await cpuTile.locator(".t").click();
await page.click("button:has-text('Propose')");
await page.waitForSelector("text=recommended", { timeout: 30000 });
await shot("05-step5-hardware-propose");
const proposeText = await text();
if (!/needs about .* GiB/.test(proposeText)) errors.push("proposal reasons missing");
await page.click("text=Next →");
await page.waitForSelector("text=6 · How hard to push");
const batchTile = page.locator(".tile:has-text('Batched requests')").first(); if (!(await batchTile.isDisabled())) errors.push('batch mode must be disabled when a chat model is in the run'); await page.locator(".tile:has-text('Concurrent requests')").first().click();
await shot("06-step6-load");
await page.click("text=Next →");
await page.waitForSelector("text=7 · Review and start");
await page.waitForSelector("text=Wall-clock time", { timeout: 30000 });
await shot("07-step7-review");
const review = await text();
if (!/Decision-grade/.test(review)) errors.push("review does not show decision-grade");

// data page
await page.goto(base + "/data");
await page.waitForSelector("details.doc", { timeout: 30000 });
await shot("08-data");

// runs
await page.goto(base + "/runs");
await page.waitForSelector("table.tbl tbody tr", { timeout: 30000 });
await page.locator("table.tbl tbody tr").first().click();
await page.waitForSelector("text=Models on hardware options", { timeout: 30000 });
await shot("09-runs");

// results: one run, then two comparable, then an incomparable pair
await page.goto(base + "/results");
await page.waitForSelector(".chip", { timeout: 30000 });
const chips = page.locator(".chips .chip");
const n = await chips.count();
const names = [];
for (let i = 0; i < n; i++) names.push(await chips.nth(i).innerText());
const pick = async (re) => { const i = names.findIndex((x) => re.test(x)); if (i >= 0) await chips.nth(i).click(); return i >= 0; };
await pick(/^smoke-m1-t1-t2-20261109T205743Z/);
await page.waitForSelector("text=Verdict.", { timeout: 60000 });
await shot("10-results-one");
const one = await text();
if (!/To match one/.test(one)) errors.push("verdict sentence missing");
// add a DSPM run: must be blocked as a different use case
await pick(/^dspm-smoke-m1-20261110T113338Z/);
await page.waitForSelector("text=Not comparable.", { timeout: 60000 });
await shot("11-results-blocked");
// swap to two DSPM runs: comparable
await pick(/^smoke-m1-t1-t2-20261109T205743Z/);
await pick(/^dspm-smoke-m1-20261110T083547Z/);
await page.waitForSelector("text=Comparable.", { timeout: 60000 });
await shot("12-results-compare-two");
// dark mode
await page.click("text=Dark mode");
await page.waitForTimeout(300);
await shot("13-results-dark");

await browser.close();
if (errors.length) { console.error("FAILED\n" + errors.join("\n")); process.exit(1); }
console.log("OK: all pages rendered without console errors; screenshots in webui/.shots/");
