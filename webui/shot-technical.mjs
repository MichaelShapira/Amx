// Screenshots of the Technical details and Setup pages in dark mode, failing on console errors or
// failed API calls. Run with the local API up:  node shot-technical.mjs [http://127.0.0.1:8601]
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
await page.goto(base + "/technical");
await page.evaluate(() => { localStorage.setItem("theme", "dark"); });
await page.goto(base + "/technical");
await page.waitForSelector("text=Hardware options and their containers", { timeout: 30000 });
await page.waitForSelector("img[alt='Architecture diagram']", { timeout: 30000 });
const natural = await page.locator("img[alt='Architecture diagram']").evaluate((el) => el.naturalWidth);
if (!natural) errors.push("architecture diagram did not render (naturalWidth 0)");
const helps = await page.locator("table.tbl th .help").count();
if (helps < 10) errors.push(`expected a help icon on every table column, found ${helps}`);
const dense = await page.locator("details:has-text('Technical specifics')").count();
if (dense < 10) errors.push(`expected the dense text behind 'Technical specifics' toggles, found ${dense}`);
await page.screenshot({ path: ".shots/technical-dark.png", fullPage: true });
// hover a header help icon so the tooltip is in the shot
await page.locator("table.tbl th .help").first().hover();
await page.waitForTimeout(200);
await page.screenshot({ path: ".shots/technical-dark-tooltip.png", clip: { x: 0, y: 0, width: 1440, height: 1000 } });
await page.goto(base + "/setup");
await page.waitForSelector("text=Shipped with the benchmark", { timeout: 30000 });
await page.locator("th:has-text('Serving software') .help").first().hover();
await page.waitForTimeout(200);
await page.screenshot({ path: ".shots/setup-dark.png", fullPage: true });
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("ok: .shots/technical-dark.png, technical-dark-tooltip.png, setup-dark.png");
