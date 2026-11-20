// Results page for one run with a load and a pricing basis set, dark mode; fails on console errors or failed API calls.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
const base = process.argv[2] ?? "http://127.0.0.1:8601";
const run = process.argv[3];
mkdirSync(".shots", { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(`console: ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("response", (r) => { if (r.url().includes("/api/") && r.status() >= 400) errors.push(`api ${r.status()} ${r.url()}`); });
await page.goto(base + "/results?runs=" + run);
await page.evaluate(() => localStorage.setItem("theme", "dark"));
await page.goto(base + "/results?runs=" + run);
await page.waitForSelector("text=Your load and your prices", { timeout: 90000 });
await page.locator("button.chip:has-text('Service: 600')").click();
await page.selectOption("select", "compute_1y");
await page.waitForSelector("text=At your load: 600 requests per minute", { timeout: 90000 });
await page.waitForTimeout(1500);
const n = await page.locator("text=Cheapest CPU option").count();
if (n < 1) errors.push("no at-your-load card");
await page.screenshot({ path: ".shots/results-dark.png", fullPage: true });
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("ok: .shots/results-dark.png");
