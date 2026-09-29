// Reviewer evidence: real Canvas frontend and live public OpenRouter catalog.
// Only backend provider/connection discovery is fixture-backed. No inference.
import { chromium, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
const out = new URL('./phase-two/', import.meta.url);
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome' });
try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 1050 } });
  await context.addInitScript(() => {
    localStorage.setItem('openhands-onboarded', '1');
    localStorage.setItem('openhands-telemetry-consent', 'denied');
    localStorage.setItem('openhands-telemetry-first-use', 'false');
    localStorage.setItem('openhands-backends', JSON.stringify([{ id: 'default-local', name: 'Local', host: location.origin, apiKey: '', kind: 'local' }]));
    localStorage.setItem('openhands-active-backend', JSON.stringify({ backendId: 'default-local', orgId: null }));
    const original = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url.includes('/api/llm/provider-connections')) {
        return Response.json([{ id: 'fixture-openrouter', display_name: 'OpenRouter connection (fixture)', provider: 'openrouter', api_key_set: true, base_url: null, created_at: 0, updated_at: 0 }]);
      }
      const response = await original(input, init);
      if (url.includes('/api/llm/providers')) {
        const body = await response.json();
        if (!body.providers.includes('openrouter')) body.providers.push('openrouter');
        return Response.json(body);
      }
      return response;
    };
  });
  const page = await context.newPage();
  const network = [];
  page.on('request', request => {
    if (request.url().startsWith('https://openrouter.ai/api/v1/models')) network.push({ url: request.url(), hasAuthorization: !!request.headers().authorization });
  });
  await page.goto('http://127.0.0.1:3001/settings/llm');
  const consent = page.getByTestId('telemetry-consent-form');
  try { await consent.waitFor({ timeout: 3000 }); await consent.getByRole('button', {name:'Confirm preferences'}).click(); } catch {}
  await page.getByTestId('add-llm-profile').click();
  const connection = page.getByTestId('llm-provider-connection-input');
  await connection.click();
  await page.getByRole('option', { name: 'OpenRouter connection (fixture)' }).click();
  await expect(page.getByTestId('llm-provider-input')).toBeDisabled();
  await expect(page.getByTestId('llm-provider-input')).toHaveValue('OpenRouter');
  await expect(page.getByTestId('save-profile-btn')).toBeDisabled();
  await page.screenshot({path: new URL('connection-selected.png', out).pathname, fullPage: true});
  const model = page.getByTestId('llm-model-input');
  await model.click();
  await model.fill('openrouter/auto');
  await page.getByRole('option', {name:'openrouter/auto', exact:true}).click();
  await page.getByTestId('profile-editor-title').click();
  await expect(page.getByRole('listbox')).toHaveCount(0);
  await expect(page.getByTestId('model-capabilities')).toBeVisible();
  await expect(page.getByTestId('save-profile-btn')).toBeEnabled();
  await expect(page.getByTestId('model-pricing')).toHaveCount(0);
  await page.screenshot({path: new URL('model-capabilities.png', out).pathname, fullPage:true});
  const facts = { network, provider: await page.getByTestId('llm-provider-input').inputValue(), providerDisabled: await page.getByTestId('llm-provider-input').isDisabled(), model: await model.inputValue(), capabilities: await page.getByTestId('model-capabilities').innerText(), showAllInitially: await page.getByTestId('openrouter-show-all-models-toggle').getByRole('checkbox').isChecked(), saveEnabled: await page.getByTestId('save-profile-btn').isEnabled() };
  await page.getByTestId('openrouter-show-all-models-toggle').getByRole('checkbox').check();
  facts.showAllAfterToggle = await page.getByTestId('openrouter-show-all-models-toggle').getByRole('checkbox').isChecked();
  await writeFile(new URL('findings.json', out), JSON.stringify(facts,null,2));
  console.log(JSON.stringify(facts,null,2));
} finally { await browser.close(); }
