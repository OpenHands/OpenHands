import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  classifySkillsSource,
  collectSkillsSourcesFromArgv,
  parseSkillMarkdown,
  parseSkillsSourcesEnv,
  resolveExternalSkillsSources,
  sourceIdFor,
  writeExternalSkillsManifest,
} from "../../scripts/skills-sources.mjs";

let workDir: string;

beforeEach(() => {
  workDir = mkdtempSync(join(tmpdir(), "skills-sources-"));
});

afterEach(() => {
  rmSync(workDir, { recursive: true, force: true });
});

function writeSkill(
  dir: string,
  { name, body }: { name?: string; body?: string } = {},
): string {
  mkdirSync(dir, { recursive: true });
  const file = join(dir, "SKILL.md");
  const frontmatter =
    name === undefined
      ? ""
      : `---\nname: ${name}\ndescription: Test skill ${name}.\n---\n\n`;
  writeFileSync(file, `${frontmatter}# Instructions\n\n${body ?? "Do it."}\n`);
  return file;
}

function collectWarnings() {
  const warnings: string[] = [];
  return {
    warnings,
    warn: (message: string) => warnings.push(message),
  };
}

describe("collectSkillsSourcesFromArgv", () => {
  it("collects repeatable --skills values in both forms", () => {
    expect(
      collectSkillsSourcesFromArgv([
        "--port",
        "9000",
        "--skills",
        "github.com/acme/skills",
        "--skills=/tmp/local-skills",
        "--skills",
        "git@github.com:acme/more.git",
      ]),
    ).toEqual([
      "github.com/acme/skills",
      "/tmp/local-skills",
      "git@github.com:acme/more.git",
    ]);
  });

  it("returns nothing when the flag is absent", () => {
    expect(collectSkillsSourcesFromArgv(["--port", "9000"])).toEqual([]);
  });
});

describe("parseSkillsSourcesEnv", () => {
  it("splits on commas and newlines and drops empties", () => {
    expect(
      parseSkillsSourcesEnv(
        " github.com/acme/skills ,\n/tmp/local-skills\n\n,git@x:y/z.git ",
      ),
    ).toEqual(["github.com/acme/skills", "/tmp/local-skills", "git@x:y/z.git"]);
  });

  it("tolerates missing or blank values", () => {
    expect(parseSkillsSourcesEnv(undefined)).toEqual([]);
    expect(parseSkillsSourcesEnv("   ")).toEqual([]);
  });
});

describe("classifySkillsSource", () => {
  it("treats an existing directory as a local source", () => {
    const dir = join(workDir, "acme-skills");
    mkdirSync(dir);
    expect(classifySkillsSource(dir)).toEqual({ kind: "local", dir });
  });

  it("resolves relative directories against the cwd", () => {
    mkdirSync(join(workDir, "rel-skills"));
    expect(classifySkillsSource("rel-skills", { cwd: workDir })).toEqual({
      kind: "local",
      dir: join(workDir, "rel-skills"),
    });
  });

  it("rejects a path that exists but is not a directory", () => {
    const file = join(workDir, "not-a-dir");
    writeFileSync(file, "x");
    const result = classifySkillsSource(file);
    if (result.kind !== "invalid") throw new Error("expected invalid");
    expect(result.reason).toContain("not a directory");
  });

  it("classifies git URL shapes", () => {
    for (const source of [
      "https://github.com/acme/skills.git",
      "ssh://git@github.com/acme/skills",
      "git@github.com:acme/skills.git",
      "file:///srv/skills.git",
    ]) {
      expect(classifySkillsSource(source)).toEqual({
        kind: "git",
        url: source,
      });
    }
  });

  it("classifies host/path shorthand as an https git URL", () => {
    expect(classifySkillsSource("github.com/acmecorp/skills")).toEqual({
      kind: "git",
      url: "https://github.com/acmecorp/skills",
    });
    expect(classifySkillsSource("gitlab.com/platform/team-skills")).toEqual({
      kind: "git",
      url: "https://gitlab.com/platform/team-skills",
    });
  });

  it("lets an existing directory win over the shorthand pattern", () => {
    // A directory literally named like a host must not be cloned.
    mkdirSync(join(workDir, "example.com"));
    expect(
      classifySkillsSource("example.com/skills", { cwd: workDir }).kind,
    ).toBe("git");
    expect(classifySkillsSource("example.com", { cwd: workDir }).kind).toBe(
      "local",
    );
  });

  it("rejects values that are neither an existing path nor a git source", () => {
    const result = classifySkillsSource("not/a/real/path", { cwd: workDir });
    if (result.kind !== "invalid") throw new Error("expected invalid");
    expect(result.reason).toContain("not/a/real/path");
  });
});

describe("sourceIdFor", () => {
  it("is stable, readable, and collision-safe", () => {
    const id = sourceIdFor("git", "https://github.com/acme/skills.git");
    expect(id).toMatch(/^skills-[0-9a-f]{8}$/);
    expect(id).toBe(sourceIdFor("git", "https://github.com/acme/skills.git"));
    // Different canonical sources never share an id even when slugs match.
    expect(sourceIdFor("git", "https://github.com/acme/skills")).not.toBe(id);
    expect(sourceIdFor("local", "/srv/skills")).not.toBe(
      sourceIdFor("git", "https://github.com/acme/skills.git"),
    );
  });
});

describe("parseSkillMarkdown", () => {
  it("parses frontmatter and body of a valid SKILL.md", () => {
    const file = join(workDir, "SKILL.md");
    writeFileSync(
      file,
      [
        "---",
        "name: my-skill",
        "description: Does the thing.",
        "triggers: [/run, /go]",
        "allowed-tools:",
        "  - Read",
        "  - Write",
        "license: MIT",
        "---",
        "",
        "# Body",
        "",
        "Instructions here.",
      ].join("\n"),
    );
    const { warn } = collectWarnings();
    const parsed = parseSkillMarkdown(file, "my-skill", warn);
    expect(parsed).toMatchObject({
      name: "my-skill",
      description: "Does the thing.",
      triggers: ["/run", "/go"],
      allowedTools: ["Read", "Write"],
      license: "MIT",
    });
    expect(parsed?.content).toContain("Instructions here.");
  });

  it("falls back to the directory name when frontmatter has no name", () => {
    const file = writeSkill(join(workDir, "unnamed"), { name: undefined });
    // writeSkill with no name still writes a body; strip frontmatter entirely.
    writeFileSync(file, "# Just a body\n\nNo frontmatter at all.\n");
    const parsed = parseSkillMarkdown(file, "unnamed", () => {});
    expect(parsed?.name).toBe("unnamed");
  });

  it("rejects a frontmatter name that differs from the expected name", () => {
    const file = writeSkill(join(workDir, "renamed"), { name: "other-name" });
    const { warnings, warn } = collectWarnings();
    expect(parseSkillMarkdown(file, "renamed", warn)).toBeNull();
    expect(warnings[0]).toContain("does not match");
    expect(warnings[0]).toContain("other-name");
  });

  it("rejects names outside the Agent Skills pattern", () => {
    const file = writeSkill(join(workDir, "bad"), { name: "Bad_Name" });
    const { warnings, warn } = collectWarnings();
    expect(parseSkillMarkdown(file, "Bad_Name", warn)).toBeNull();
    expect(warnings[0]).toContain("not a valid Agent Skills name");
  });

  it("rejects an unclosed frontmatter fence and an empty body", () => {
    const unclosed = join(workDir, "unclosed.md");
    writeFileSync(unclosed, "---\nname: x\n\nno closing fence\n");
    const { warnings, warn } = collectWarnings();
    expect(parseSkillMarkdown(unclosed, "x", warn)).toBeNull();
    expect(warnings[0]).toContain("never closes");

    const empty = join(workDir, "empty.md");
    writeFileSync(empty, "---\nname: x\ndescription: d\n---\n\n");
    expect(parseSkillMarkdown(empty, "x", warn)).toBeNull();
    expect(warnings[1]).toContain("no instructions body");
  });

  it("keeps a skill with no description but warns about it", () => {
    const file = join(workDir, "nodesc.md");
    writeFileSync(file, "---\nname: nodesc\n---\n\nBody.\n");
    const { warnings, warn } = collectWarnings();
    const parsed = parseSkillMarkdown(file, "nodesc", warn);
    expect(parsed?.name).toBe("nodesc");
    expect(parsed?.description).toBeNull();
    expect(warnings[0]).toContain("no description");
  });
});

describe("resolveExternalSkillsSources", () => {
  it("finds skills in every supported layout of a local source", () => {
    const source = join(workDir, "acme-skills");
    writeSkill(join(source, "skills", "from-skills-dir"), {
      name: "from-skills-dir",
    });
    writeSkill(join(source, ".agents", "skills", "from-agents"), {
      name: "from-agents",
    });
    writeSkill(join(source, "microagents", "from-micro"), {
      name: "from-micro",
    });
    writeSkill(join(source, "bare-dir"), { name: "bare-dir" });

    const { warnings, warn } = collectWarnings();
    const manifest = resolveExternalSkillsSources([source], {
      cacheDir: join(workDir, "cache"),
      warn,
    });

    expect(warnings).toEqual([]);
    expect(manifest.sources).toHaveLength(1);
    expect(manifest.sources[0]).toMatchObject({
      source,
      kind: "local",
      path: source,
    });
    const names = manifest.skills.map((skill) => skill.name).sort();
    expect(names).toEqual([
      "bare-dir",
      "from-agents",
      "from-micro",
      "from-skills-dir",
    ]);
    const entry = manifest.skills.find(
      (skill) => skill.name === "from-skills-dir",
    );
    expect(entry?.source_id).toBe(manifest.sources[0].id);
    expect(entry?.path).toBe(
      join(source, "skills", "from-skills-dir", "SKILL.md"),
    );
  });

  it("treats a source directory that is itself a skill", () => {
    const source = join(workDir, "single-skill");
    writeSkill(source, { name: "single-skill" });
    const manifest = resolveExternalSkillsSources([source], {
      cacheDir: join(workDir, "cache"),
      warn: () => {},
    });
    expect(manifest.skills.map((skill) => skill.name)).toEqual([
      "single-skill",
    ]);
  });

  it("warns and keeps going when a source fails", () => {
    const good = join(workDir, "good-skills");
    writeSkill(join(good, "skills", "kept"), { name: "kept" });

    const { warnings, warn } = collectWarnings();
    const manifest = resolveExternalSkillsSources(
      [join(workDir, "missing"), good, ""],
      { cacheDir: join(workDir, "cache"), warn },
    );

    expect(manifest.skills.map((skill) => skill.name)).toEqual(["kept"]);
    expect(warnings.some((message) => message.includes("missing"))).toBe(true);
    expect(warnings.some((message) => message.includes("empty"))).toBe(true);
  });

  it("warns when a source yields no valid skills", () => {
    const empty = join(workDir, "empty-source");
    mkdirSync(empty);
    const { warnings, warn } = collectWarnings();
    const manifest = resolveExternalSkillsSources([empty], {
      cacheDir: join(workDir, "cache"),
      warn,
    });
    expect(manifest.skills).toEqual([]);
    expect(manifest.sources).toEqual([]);
    expect(warnings[0]).toContain("no valid Agent Skills");
  });

  it("keeps the first of two same-named skills in one source", () => {
    const source = join(workDir, "dupes");
    writeSkill(join(source, "skills", "same"), { name: "same" });
    writeSkill(join(source, "microagents", "same"), { name: "same" });
    const { warnings, warn } = collectWarnings();
    const manifest = resolveExternalSkillsSources([source], {
      cacheDir: join(workDir, "cache"),
      warn,
    });
    expect(manifest.skills.map((skill) => skill.name)).toEqual(["same"]);
    expect(warnings[0]).toContain("duplicate skill name");
  });

  it("clones a git source and resolves a repo that is itself one skill", () => {
    // A single-skill repository: the clone lands in a synthetic
    // `<slug>-<hash>` cache dir, so the frontmatter name is checked against
    // the repository name — without that substitution this layout could
    // never validate.
    const repo = join(workDir, "git-skill");
    writeSkill(repo, { name: "git-skill" });
    spawnSync("git", ["init", "-q"], { cwd: repo });
    spawnSync("git", ["add", "-A"], { cwd: repo });
    spawnSync(
      "git",
      ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
      { cwd: repo },
    );

    const { warnings, warn } = collectWarnings();
    const manifest = resolveExternalSkillsSources([`file://${repo}`], {
      cacheDir: join(workDir, "cache"),
      warn,
    });

    expect(warnings).toEqual([]);
    expect(manifest.skills.map((skill) => skill.name)).toEqual(["git-skill"]);
    expect(manifest.sources[0].kind).toBe("git");
    // The resolved path points at the cached clone's SKILL.md.
    expect(manifest.skills[0].path).toContain(
      join("cache", manifest.sources[0].id),
    );
  });

  it("warns instead of failing when a git clone fails", () => {
    const { warnings, warn } = collectWarnings();
    const manifest = resolveExternalSkillsSources(
      [`file://${join(workDir, "no-such-repo")}`],
      { cacheDir: join(workDir, "cache"), warn },
    );
    expect(manifest.skills).toEqual([]);
    expect(warnings[0]).toContain("git clone failed");
  });
});

describe("writeExternalSkillsManifest", () => {
  it("writes the manifest the serving layer injects", () => {
    const file = join(workDir, "nested", "manifest.json");
    const manifest = { version: 1, sources: [], skills: [] };
    expect(writeExternalSkillsManifest(file, manifest)).toBe(file);
    expect(JSON.parse(readFileSync(file, "utf8"))).toEqual(manifest);
    expect(existsSync(file)).toBe(true);
  });
});
