import { chromium } from "playwright";
const base = process.argv[2] ?? "http://127.0.0.1:8601"; const run = process.argv[3];
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => { if (m.type() === "error") errors.push(`console: ${m.text()}`); });
page.on("response", (r) => { if (r.url().includes("/api/") && r.status() >= 400) errors.push(`api ${r.status()} ${r.url()}`); });
await page.goto(base + "/results?runs=" + run);
await page.evaluate(() => localStorage.setItem("theme", "dark"));
await page.goto(base + "/results?runs=" + run);
await page.waitForSelector("text=Where the CPU is the better choice", { timeout: 90000 });
await page.waitForTimeout(800);
const el = page.locator("h2:has-text('Where the CPU is the better choice')").locator("..");
await el.screenshot({ path: ".shots/where-cpu-wins.png" });
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("ok");
