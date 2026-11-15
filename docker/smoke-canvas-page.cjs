// Open the Canvas pages of a running agent-canvas image in a browser and fail
// when one renders nothing. curl only sees the HTML shell, which answers 200
// even when the router matches no route: in 1.36.0 the basename became
// "/canvas/", so "/canvas" (where sign-in lands) rendered a blank page (#19405).
//
// Usage: node docker/smoke-canvas-page.cjs <origin> <base-path>
// Needs the `playwright` package and Chromium; the Docker workflow installs
// them and points NODE_PATH at them.
const { chromium } = require("playwright");

const [origin, basePath] = process.argv.slice(2);
if (!origin || !basePath) {
  console.error(
    "usage: node docker/smoke-canvas-page.cjs <origin> <base-path>",
  );
  process.exit(2);
}

const base = basePath.replace(/\/+$/, "");
// The bare base path, with a trailing slash, and with a query like sign-in's.
const paths = [base, `${base}/`, `${base}?login_method=smoke`];

async function check(browser, path) {
  const page = await browser.newPage();
  const routerWarnings = [];
  page.on("console", (message) => {
    if (/is not able to match the URL/.test(message.text())) {
      routerWarnings.push(message.text());
    }
  });
  try {
    await page.goto(`${origin}${path}`, { waitUntil: "load", timeout: 60_000 });
    const rendered = await page
      .waitForFunction(() => document.body.innerText.trim().length > 0, null, {
        timeout: 30_000,
      })
      .then(() => true)
      .catch(() => false);
    const problems = [];
    if (!rendered) problems.push("the page rendered no text");
    if (routerWarnings.length > 0) problems.push(routerWarnings[0]);
    const where = new URL(page.url()).pathname;
    console.log(`${path} -> ${where}: ${problems.length ? "FAIL" : "ok"}`);
    return problems.map((problem) => `${path}: ${problem}`);
  } finally {
    await page.close();
  }
}

(async () => {
  const browser = await chromium.launch();
  const failures = [];
  try {
    for (const path of paths) failures.push(...(await check(browser, path)));
  } finally {
    await browser.close();
  }
  for (const failure of failures) console.log(`::error::${failure}`);
  process.exit(failures.length ? 1 : 0);
})();
