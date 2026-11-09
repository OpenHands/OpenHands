// Real production app + real Agent Server. No routes, responses, or models mocked.
// Usage: node .pr/issue-17567/initialization/exercise.mjs RUN before|after [ROUNDS=5]
import { chromium, expect } from '@playwright/test';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const [runDir, label, roundsArgument = '5'] = process.argv.slice(2);
const rounds = Number(roundsArgument);
const run = JSON.parse(await readFile(join(runDir, 'run.json'), 'utf8'));
const fixture = JSON.parse(await readFile(join(runDir, 'evidence/fixture.json'), 'utf8'));
const output = join(runDir, 'evidence/initialization');
await mkdir(output, { recursive: true });
const browser = await chromium.launch();
const setup = await browser.newContext();
const setupPage = await setup.newPage();
await setupPage.goto(run.baseUrl);
await setupPage.waitForTimeout(1500);
if (await setupPage.getByTestId('telemetry-consent-form').isVisible()) {
  await setupPage.getByTestId('telemetry-consent-form').getByRole('checkbox').uncheck();
  await setupPage.getByTestId('confirm-telemetry-preferences').click();
}
if (await setupPage.getByTestId('onboarding-skip').isVisible()) {
  await setupPage.getByTestId('onboarding-skip').click();
}
const storageState = await setup.storageState();
await setup.close();

const summary = {
  label, revision: run.revision, buildId: run.buildId, backend: run.versions,
  fixture, trigger: 'Observe Monaco onDidCreateDiffEditor, then click the real Planner control in requestAnimationFrame.',
  results: [],
};
try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    const size = `${viewport.width}x${viewport.height}`;
    const context = await browser.newContext({ viewport, storageState, recordVideo: { dir: output, size: viewport } });
    const page = await context.newPage();
    const video = page.video();
    const errors = [];
    page.on('pageerror', error => errors.push({ type: 'pageerror', message: error.message }));
    page.on('console', message => {
      if (message.type() === 'error') errors.push({ type: 'console', message: message.text() });
    });
    const tab = async name => {
      const button = page.getByTestId(`conversation-tab-${name}`);
      if (await button.isVisible()) {
        if (await button.getAttribute('aria-selected') !== 'true') await button.click();
      } else {
        await page.getByRole('button', { name: 'Customize tabs' }).click();
        await page.getByTestId(`conversation-tabs-menu-open-${name}`).click();
      }
      await expect(button).toHaveAttribute('aria-selected', 'true');
    };
    const changes = async () => {
      await tab('commits');
      const toggle = page.getByTestId('uncommitted-changes-row-toggle');
      await expect(toggle).toBeVisible();
      if (await toggle.getAttribute('aria-expanded') !== 'true') await toggle.click();
    };
    const row = page.getByTestId('uncommitted-changes-row-content').getByTestId('file-diff-viewer-outer').filter({ hasText: 'src/calc.py' });
    const contents = async () => {
      await expect(row.locator('.monaco-diff-editor')).toBeVisible();
      await expect(row.getByTestId('editor-container')).toContainText('def sub');
      await expect(row.getByTestId('editor-container')).toContainText('return a - b');
      // Let the existing accordion fade finish so the evidence remains readable.
      await page.waitForTimeout(250);
    };
    try {
      await page.goto(`${run.baseUrl}/conversations/${fixture.conversationId}${viewport.width < 1024 ? '/panel' : ''}`);
      if (viewport.width >= 1024) await page.getByTestId('right-panel-toggle').click();
      await changes();
      await row.getByTestId('collapse').click();
      await contents();
      await page.screenshot({ path: join(output, `${label}-${size}-diff.png`) });
      await tab('planner');
      await expect(page.getByRole('heading', { name: 'Diff lifecycle verification' })).toBeVisible();
      errors.length = 0;
      await page.evaluate(() => {
        window.initializationProbe = { mode: 'ordinary', round: 0, events: [] };
        window.monaco.editor.onDidCreateDiffEditor(editor => {
          const probe = window.initializationProbe;
          probe.events.push({ phase: 'created', round: probe.round, mode: probe.mode, time: performance.now() });
          if (probe.mode === 'raf') requestAnimationFrame(() => {
            probe.events.push({ phase: 'planner-click', round: probe.round, mode: probe.mode, hasModel: !!editor.getModel(), time: performance.now() });
            document.querySelector('[data-testid="conversation-tab-planner"]').click();
          });
        });
      });
      for (const mode of ['raf', 'ordinary']) {
        for (let round = 1; round <= rounds; round += 1) {
          await changes();
          await expect(page.getByTestId('conversation-tab-planner')).toBeVisible();
          await page.evaluate(({ mode, round }) => Object.assign(window.initializationProbe, { mode, round }), { mode, round });
          const errorStart = errors.length;
          await row.getByTestId('collapse').click();
          if (mode === 'ordinary') {
            await contents();
            await tab('planner');
          }
          await expect(page.getByTestId('conversation-tab-planner')).toHaveAttribute('aria-selected', 'true');
          await expect(page.locator('.monaco-diff-editor')).toHaveCount(0);
          await expect(page.getByRole('heading', { name: 'Diff lifecycle verification' })).toBeVisible();
          // Monaco reports disposal errors asynchronously. Let that queue drain.
          await page.waitForTimeout(80);
          const models = await page.evaluate(() => window.monaco.editor.getModels().map(model => ({ uri: model.uri.toString(), text: model.getValue() })));
          const result = { viewport, mode, round, errors: errors.slice(errorStart), models };
          summary.results.push(result);
          console.log(JSON.stringify(result));
        }
      }
      await page.screenshot({ path: join(output, `${label}-${size}-planner.png`) });
      await changes();
      await page.evaluate(() => { window.initializationProbe.mode = 'ordinary'; });
      await row.getByTestId('collapse').click();
      await contents();
      await page.screenshot({ path: join(output, `${label}-${size}-returned.png`) });
      await tab('planner');
      await expect(page.locator('.monaco-diff-editor')).toHaveCount(0);
      await page.waitForTimeout(80);
      const evidence = await page.evaluate(() => ({ events: window.initializationProbe.events, models: window.monaco.editor.getModels().map(model => ({ uri: model.uri.toString(), text: model.getValue() })), userAgent: navigator.userAgent, scripts: [...document.scripts].map(script => script.src).filter(url => /monaco-editor/.test(url)) }));
      await writeFile(join(output, `${label}-${size}-events.json`), JSON.stringify({ ...evidence, errors, correctContentsReturned: true }, null, 2) + '\n');
    } finally {
      await context.close();
      await video.saveAs(join(output, `${label}-${size}.webm`));
    }
  }
} finally {
  await writeFile(join(output, `${label}-results.json`), JSON.stringify(summary, null, 2) + '\n');
  await browser.close();
}
for (const result of summary.results) {
  if (result.mode === 'ordinary') expect(result.errors).toEqual([]);
  if (label !== 'before') {
    expect(result.errors.filter(error => error.message.includes('TextModel got disposed before DiffEditorWidget model got reset'))).toEqual([]);
  }
  expect(result.models).toEqual([]);
}
if (label === 'before') {
  // A frame can land after initialization. Require a reproduction at each size,
  // while retaining every attempt rather than selecting only failing frames.
  for (const width of [1440, 390]) {
    expect(summary.results.some(result => result.viewport.width === width && result.mode === 'raf' && result.errors.some(error => error.message.includes('TextModel got disposed before DiffEditorWidget model got reset')))).toBe(true);
  }
}
