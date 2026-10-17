import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1400, height: 1000 },
  deviceScaleFactor: 2,
});
await page.goto(
  "file:///workspace/project/OpenHands/.pr/18445-browser-evidence.html",
);
await page.waitForTimeout(500);
await page.screenshot({
  path: "/workspace/project/OpenHands/.pr/18445-browser-evidence.png",
  fullPage: true,
});
await browser.close();
console.log("shot done");
