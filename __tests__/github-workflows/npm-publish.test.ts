// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

describe("npm publish workflow", () => {
  it("builds both package surfaces with the production PostHog key", () => {
    const workflow = read(".github/workflows/npm-publish.yml");
    const buildAppStep = workflow.match(
      /- name: Build app[\s\S]*?(?=\n\s*- name: Build library)/,
    )?.[0];
    const buildLibraryStep = workflow.match(
      /- name: Build library[\s\S]*?(?=\n\s*- name: Verify package contents)/,
    )?.[0];

    expect(buildAppStep).toBeTruthy();
    expect(buildAppStep).toContain(
      "VITE_POSTHOG_API_KEY: ${{ vars.POSTHOG_PROD_KEY }}",
    );
    expect(buildAppStep).toContain("npm run build");
    expect(buildLibraryStep).toContain(
      "VITE_POSTHOG_API_KEY: ${{ vars.POSTHOG_PROD_KEY }}",
    );
    expect(buildLibraryStep).toContain("npm run build:lib");
  });

  it("passes the selected PostHog key through Docker's supported build arg", () => {
    const workflow = read(".github/workflows/docker.yml");
    const dockerfile = read("docker/Dockerfile");

    expect(workflow).toContain(
      "VITE_POSTHOG_API_KEY=${{ steps.prep.outputs.posthog_api_key }}",
    );
    expect(workflow).not.toContain("VITE_POSTHOG_CLIENT_KEY");
    expect(workflow).not.toContain("vite_app_env");
    expect(dockerfile).toContain('ARG VITE_POSTHOG_API_KEY=""');
    expect(dockerfile).toContain(
      "ENV VITE_POSTHOG_API_KEY=${VITE_POSTHOG_API_KEY}",
    );
    expect(dockerfile).not.toContain("VITE_APP_ENV");
  });
});
