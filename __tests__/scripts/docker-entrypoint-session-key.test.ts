/** @vitest-environment node */
import { describe, it, expect } from 'vitest';
import { execFileSync } from 'node:child_process';
import { readFileSync, mkdtempSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

function getBashExecutable(): string {
  if (process.platform !== 'win32') {
    return 'bash';
  }
  const possiblePaths = [
    'C:\\Program Files\\Git\\bin\\bash.exe',
    'C:\\Program Files\\Git\\usr\\bin\\bash.exe',
    'C:\\Program Files (x86)\\Git\\bin\\bash.exe',
    `${process.env.LOCALAPPDATA}\\Programs\\Git\\bin\\bash.exe`,
    `${process.env.PROGRAMFILES}\\Git\\bin\\bash.exe`,
  ];
  for (const p of possiblePaths) {
    if (existsSync(p)) {
      return p;
    }
  }
  try {
    const whereOutput = execFileSync('where.exe', ['bash'], { encoding: 'utf-8' }).trim();
    const firstLine = whereOutput.split(/\r?\n/)[0];
    if (firstLine && existsSync(firstLine)) {
      return firstLine;
    }
  } catch {
    // fallback if where.exe fails
  }
  return 'bash';
}

describe('docker/entrypoint.sh session API key resolution', () => {
  const entrypointPath = resolve(__dirname, '../../docker/entrypoint.sh');
  const entrypointContent = readFileSync(entrypointPath, 'utf-8');

  // Extract the session API key resolution block
  const match = entrypointContent.match(
    /# >>> session-api-key-config([\s\S]*?)# <<< session-api-key-config/
  );
  if (!match) {
    throw new Error('Could not find session-api-key-config markers in docker/entrypoint.sh');
  }
  const extractedScript = match[1];

  const runResolution = (envOverrides: {
    LOCAL_BACKEND_API_KEY?: string;
    OH_SESSION_API_KEYS_0?: string;
  }) => {
    const tempDir = mkdtempSync(join(tmpdir(), 'oh-test-'));
    const scriptFile = join(tempDir, 'test-entrypoint.sh');
    try {
      const bashEnvSetup = [
        envOverrides.LOCAL_BACKEND_API_KEY !== undefined
          ? `export LOCAL_BACKEND_API_KEY="${envOverrides.LOCAL_BACKEND_API_KEY}"`
          : 'unset LOCAL_BACKEND_API_KEY',
        envOverrides.OH_SESSION_API_KEYS_0 !== undefined
          ? `export OH_SESSION_API_KEYS_0="${envOverrides.OH_SESSION_API_KEYS_0}"`
          : 'unset OH_SESSION_API_KEYS_0',
      ].join('\n');

      const scriptContent = `#!/usr/bin/env bash
set -uo pipefail
log() { :; }
STATE_DIR="${tempDir.replace(/\\/g, '/')}"
${bashEnvSetup}
${extractedScript}
echo "RESULT=$OH_SESSION_API_KEYS_0"
`;

      writeFileSync(scriptFile, scriptContent, 'utf-8');

      const bashBin = getBashExecutable();
      const output = execFileSync(bashBin, [scriptFile], {
        encoding: 'utf-8',
      });

      const resultMatch = output.match(/RESULT=(.*)/);
      return resultMatch ? resultMatch[1].trim() : '';
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  };

  it('exports OH_SESSION_API_KEYS_0 when only LOCAL_BACKEND_API_KEY is supplied', () => {
    const key = runResolution({
      LOCAL_BACKEND_API_KEY: 'custom-secret-key',
    });
    expect(key).toBe('custom-secret-key');
  });

  it('preserves existing OH_SESSION_API_KEYS_0 when both are provided', () => {
    const key = runResolution({
      LOCAL_BACKEND_API_KEY: 'fallback-secret',
      OH_SESSION_API_KEYS_0: 'primary-session-key',
    });
    expect(key).toBe('primary-session-key');
  });

  it('exports OH_SESSION_API_KEYS_0 when only OH_SESSION_API_KEYS_0 is provided', () => {
    const key = runResolution({
      OH_SESSION_API_KEYS_0: 'direct-session-key',
    });
    expect(key).toBe('direct-session-key');
  });

  it('generates a non-empty key when neither is supplied', () => {
    const key = runResolution({});
    expect(key).toBeTruthy();
    expect(key.length).toBeGreaterThan(16);
  });
});