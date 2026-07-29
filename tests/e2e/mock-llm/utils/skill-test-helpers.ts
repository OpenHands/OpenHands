/**
 * Helpers for skill loading E2E tests.
 *
 * File-system operations (create/remove SKILL.md files) and API
 * assertions (verify activated_skills on events) are separated here
 * to avoid type-resolution conflicts between node built-in imports
 * and @playwright/test types in the same file (TypeScript 6 / Node 24).
 *
 * Docker support: When running against a Docker container, the agent-server
 * filesystem is isolated from the host. The Playwright config sets env vars
 * for the container-side paths so the test can register the correct paths
 * with the agent-server while creating files on the host (volume-mounted).
 */

import { resolve, join } from "path";
import { mkdirSync, writeFileSync, rmSync, existsSync } from "fs";
import { execSync } from "child_process";
import { homedir } from "os";

// ── Paths ────────────────────────────────────────────────────────────

/** STATE_DIR matches playwright.mock-llm.config.ts */
export const STATE_DIR = resolve(".tmp/mock-llm-state");

/**
 * Root directory for skill-test workspace git repos (HOST-side).
 * Each call to `createProjectSkillRepo` creates a self-contained git repo
 * here with the skill file already committed, so the agent-server's
 * worktree machinery picks it up (worktrees only contain committed content).
 */
export const SKILL_REPOS_DIR = resolve(".tmp/mock-llm-skill-repos");

/**
 * The path the agent-server sees for skill repos.
 * In npm mode this is the same as SKILL_REPOS_DIR (same filesystem).
 * In Docker mode this is the container-side mount point set by the config.
 */
export const SKILL_REPOS_AGENT_DIR =
  process.env.MOCK_LLM_SKILL_REPOS_CONTAINER_DIR ?? SKILL_REPOS_DIR;

/**
 * User-level skills directory — HOST-side (for file creation/removal).
 * In Docker mode, we use a local temp dir that is volume-mounted into the
 * container at the agent-server's expected `~/.openhands/skills/` path.
 */
export const USER_SKILLS_DIR = process.env.MOCK_LLM_USER_SKILLS_HOST_DIR
  ? resolve(process.env.MOCK_LLM_USER_SKILLS_HOST_DIR)
  : join(homedir(), ".openhands", "skills");

// ── Skill content builders ───────────────────────────────────────────

function makeSkillMd(
  name: string,
  trigger: string,
  description: string,
): string {
  return [
    "---",
    `name: ${name}`,
    `description: ${description}`,
    "triggers:",
    `- ${trigger}`,
    "---",
    "",
    `This is the ${name} skill content for E2E testing.`,
    `It should activate when the keyword "${trigger}" appears.`,
  ].join("\n");
}

/**
 * Create a standalone git repo with a project skill committed.
 *
 * The agent-server creates a git worktree for each conversation from the
 * source workspace. Only committed files appear in worktrees, so the skill
 * must be committed to the repo for `load_project_skills` to find it.
 *
 * @returns Object with `hostDir` (absolute host path for file ops) and
 *          `agentDir` (path the agent-server sees — same in npm mode,
 *          container-side mount in Docker mode).
 */
export function createProjectSkillRepo(
  name: string,
  trigger: string,
  description = "E2E test skill",
): { hostDir: string; agentDir: string } {
  const repoDir = join(SKILL_REPOS_DIR, `${name}-repo`);
  // Start fresh each time
  rmSync(repoDir, { recursive: true, force: true });
  mkdirSync(repoDir, { recursive: true });

  // Write the skill file
  const skillDir = join(repoDir, ".agents", "skills", name);
  mkdirSync(skillDir, { recursive: true });
  writeFileSync(
    join(skillDir, "SKILL.md"),
    makeSkillMd(name, trigger, description),
  );

  // Initialize as a git repo and commit
  const opts = { cwd: repoDir, stdio: "pipe" as const };
  execSync("git init", opts);
  execSync('git config user.email "test@test.com"', opts);
  execSync('git config user.name "Test"', opts);
  execSync("git add -A", opts);
  execSync('git commit -m "Add project skill"', opts);

  const hostDir = resolve(repoDir);
  const agentDir = join(SKILL_REPOS_AGENT_DIR, `${name}-repo`);
  return { hostDir, agentDir };
}

/** Remove a project skill repo created by `createProjectSkillRepo`. */
export function removeProjectSkillRepo(name: string): void {
  const repoDir = join(SKILL_REPOS_DIR, `${name}-repo`);
  rmSync(repoDir, { recursive: true, force: true });
}

export function writeUserSkill(
  name: string,
  trigger: string,
  description = "E2E test user skill",
): void {
  const dir = join(USER_SKILLS_DIR, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "SKILL.md"), makeSkillMd(name, trigger, description));
}

export function removeUserSkill(name: string): void {
  const dir = join(USER_SKILLS_DIR, name);
  rmSync(dir, { recursive: true, force: true });
}

export function userSkillExists(name: string): boolean {
  return existsSync(join(USER_SKILLS_DIR, name, "SKILL.md"));
}

export function userSkillDirExists(name: string): boolean {
  return existsSync(join(USER_SKILLS_DIR, name));
}
