import { SKILLS_CATALOG } from "@openhands/extensions/skills";
import {
  buildSkillEnablementFilter,
  type SkillEnablement,
} from "#/utils/skill-enablement";

/**
 * ACP registry key for the Claude Code harness. Matches
 * ``ACP_PROVIDERS`` / ``tags.acpserver``.
 */
export const CLAUDE_CODE_ACP_SERVER = "claude-code";

/**
 * Token in Claude Code's default ACP command
 * (``npx -y @agentclientprotocol/claude-agent-acp@…``). Used as a fallback
 * when a custom command keeps the Claude adapter without the registry key.
 */
export const CLAUDE_CODE_ACP_COMMAND_TOKEN = "claude-agent-acp";

/**
 * ``AgentLaunchAdditions.system_message_suffix_append`` max length on
 * agent-server 1.46.x (Pydantic ``max_length=32768``, ``extra="forbid"``).
 * ``skills_append`` is not available until software-agent-sdk#4717.
 */
export const AGENT_LAUNCH_SUFFIX_APPEND_MAX_LENGTH = 32768;

/** Wrapper tag so Claude can tell Canvas-projected skills from its own. */
export const CANVAS_ENABLED_SKILLS_TAG = "CANVAS_ENABLED_SKILLS";

/**
 * Machine-readable marker appended when the 32 KiB launch suffix cannot hold
 * every enabled skill in full. Names are listed so eviction is not silent.
 */
export const CANVAS_SKILLS_TRUNCATED_MARKER = "CANVAS_SKILLS_TRUNCATED";

const SUFFIX_INTRO =
  "The following skill instructions were enabled in Agent Canvas for this session.";

const MIN_TRUNCATED_SKILL_CHARS = 256;

export interface ClaudeCodeAcpAgentHint {
  agentKind?: string | null;
  acpServer?: string | null;
  acpCommand?: string | readonly string[] | null;
}

/**
 * True when this launch is a Claude Code ACP agent. OpenHands and other ACP
 * harnesses (Codex, Gemini, custom) stay out of this overlay — #16905 is
 * Claude-only.
 */
export function isClaudeCodeAcpAgent(hint: ClaudeCodeAcpAgentHint): boolean {
  if (hint.agentKind != null && hint.agentKind !== "acp") return false;
  if (hint.acpServer === CLAUDE_CODE_ACP_SERVER) return true;
  const command = normalizeAcpCommand(hint.acpCommand);
  return command.includes(CLAUDE_CODE_ACP_COMMAND_TOKEN);
}

export function normalizeAcpCommand(
  command: string | readonly string[] | null | undefined,
): string {
  if (Array.isArray(command)) return command.join(" ");
  return typeof command === "string" ? command : "";
}

interface CatalogSkillEntry {
  name: string;
  content: string;
}

/**
 * Catalog skills this Claude ACP session should receive: the persisted
 * allow-list (with the deny-list winning), plus a slash-invoked skill for
 * this conversation only — same rule as ``buildAgentContext``.
 */
export function selectClaudeAcpCatalogSkills(
  enablement: SkillEnablement,
  invokedCatalogSkill?: string,
  catalog: readonly CatalogSkillEntry[] = SKILLS_CATALOG,
): CatalogSkillEntry[] {
  const isEnabled = buildSkillEnablementFilter(enablement);
  const selected: CatalogSkillEntry[] = [];
  const seen = new Set<string>();

  const consider = (entry: CatalogSkillEntry | undefined) => {
    if (!entry || seen.has(entry.name)) return;
    if (entry.name !== invokedCatalogSkill && !isEnabled(entry.name)) return;
    seen.add(entry.name);
    selected.push(entry);
  };

  if (invokedCatalogSkill) {
    consider(catalog.find((entry) => entry.name === invokedCatalogSkill));
  }
  for (const entry of catalog) {
    consider(entry);
  }
  return selected;
}

function formatSkillBlock(entry: CatalogSkillEntry): string {
  return `## ${entry.name}\n${entry.content.trim()}`;
}

function wrapSkillSuffix(body: string): string {
  return `<${CANVAS_ENABLED_SKILLS_TAG}>\n${SUFFIX_INTRO}\n\n${body}\n</${CANVAS_ENABLED_SKILLS_TAG}>`;
}

function wrapOverhead(): number {
  return wrapSkillSuffix("").length;
}

function estimatePackedBodyLength(
  skills: readonly CatalogSkillEntry[],
): number {
  if (skills.length === 0) return 0;
  let total = 0;
  for (const entry of skills) {
    total += formatSkillBlock(entry).length;
  }
  return total + Math.max(0, skills.length - 1) * 2;
}

/** Worst-case marker length if every skill is both truncated and omitted. */
function maxMarkerReserve(names: readonly string[]): number {
  if (names.length === 0) return 0;
  return (
    formatTruncationMarker({ truncated: [...names], omitted: [...names] })
      .length + 2
  ); // leading "\n\n"
}

function formatTruncationMarker(parts: {
  truncated: readonly string[];
  omitted: readonly string[];
}): string {
  const attrs: string[] = [];
  if (parts.truncated.length > 0) {
    attrs.push(`truncated="${parts.truncated.join(",")}"`);
  }
  if (parts.omitted.length > 0) {
    attrs.push(`omitted="${parts.omitted.join(",")}"`);
  }
  return `<!-- ${CANVAS_SKILLS_TRUNCATED_MARKER} ${attrs.join(" ")} -->`;
}

/** Skill #16905 targets — prioritized when packing under the 32 KiB cap. */
export const AUTOMATION_SKILL_NAME = "openhands-automation";

/** Sections that must survive truncation for schedule/lifecycle edits. */
const AUTOMATION_MUST_KEEP_HEADINGS = [
  "## Managing Automations",
  "## Run Lifecycle",
] as const;

/** Useful but secondary automation sections (kept when budget allows). */
const AUTOMATION_PREFERRED_HEADINGS = [
  "## Automation Creation Process",
  "## Creating Automations",
  "## Trigger Types",
  "## Authentication",
] as const;

interface MarkdownSection {
  heading: string;
  body: string;
}

/** Split markdown into leading intro + `##` sections (heading line included). */
export function splitMarkdownSections(content: string): {
  intro: string;
  sections: MarkdownSection[];
} {
  const normalized = content.replace(/\r\n/g, "\n");
  const matches = [...normalized.matchAll(/^## .+$/gm)];
  if (matches.length === 0) {
    return { intro: normalized.trimEnd(), sections: [] };
  }
  const intro = normalized.slice(0, matches[0].index).trimEnd();
  const sections: MarkdownSection[] = [];
  for (let i = 0; i < matches.length; i += 1) {
    const start = matches[i].index ?? 0;
    const end =
      i + 1 < matches.length
        ? (matches[i + 1].index ?? normalized.length)
        : normalized.length;
    const chunk = normalized.slice(start, end).trimEnd();
    const nl = chunk.indexOf("\n");
    const heading = nl === -1 ? chunk : chunk.slice(0, nl);
    const body = nl === -1 ? "" : chunk.slice(nl + 1);
    sections.push({ heading, body });
  }
  return { intro, sections };
}

function sectionBlock(section: MarkdownSection): string {
  return section.body ? `${section.heading}\n${section.body}` : section.heading;
}

function countFenceMarkers(text: string): number {
  return (text.match(/```/g) ?? []).length;
}

/**
 * Truncate ``text`` to ``maxLen`` without leaving an open fenced code block.
 * Prefers cutting on a newline; closes an odd fence with a trailing fence when
 * needed so downstream markers stay outside code.
 */
export function fenceSafeTruncate(text: string, maxLen: number): string {
  if (maxLen <= 0) return "";
  if (text.length <= maxLen) {
    if (countFenceMarkers(text) % 2 === 0) return text;
    const closer = "\n```";
    if (text.length + closer.length <= maxLen) return `${text}${closer}`;
    return fenceSafeTruncate(
      text.slice(0, Math.max(0, maxLen - closer.length)),
      maxLen,
    );
  }

  let slice = text.slice(0, maxLen);
  const lastNl = slice.lastIndexOf("\n");
  if (lastNl >= Math.floor(maxLen * 0.6)) {
    slice = slice.slice(0, lastNl);
  }

  if (countFenceMarkers(slice) % 2 === 1) {
    const closer = "\n```";
    if (slice.length + closer.length <= maxLen) {
      return `${slice}${closer}`;
    }
    const lastFence = slice.lastIndexOf("```");
    if (lastFence >= 0) {
      slice = slice.slice(0, lastFence).trimEnd();
    }
  }
  return slice;
}

/**
 * Rebuild automation skill body under ``maxLen`` by keeping schedule/lifecycle
 * sections and dropping large reference sections first (not head-slicing).
 */
export function prioritizeAutomationContent(
  content: string,
  maxLen: number,
): { content: string; truncated: boolean } {
  const trimmed = content.trim();
  if (trimmed.length <= maxLen) {
    return { content: trimmed, truncated: false };
  }

  const { intro, sections } = splitMarkdownSections(trimmed);
  if (sections.length === 0) {
    const cut = fenceSafeTruncate(trimmed, maxLen);
    return { content: cut, truncated: cut.length < trimmed.length };
  }

  const byHeading = new Map(
    sections.map((section) => [section.heading, section]),
  );
  const must = AUTOMATION_MUST_KEEP_HEADINGS.map((h) =>
    byHeading.get(h),
  ).filter((s): s is MarkdownSection => s != null);
  const preferred = AUTOMATION_PREFERRED_HEADINGS.map((h) =>
    byHeading.get(h),
  ).filter((s): s is MarkdownSection => s != null);
  const used = new Set([...must, ...preferred].map((s) => s.heading));
  const deferred = sections
    .filter((s) => !used.has(s.heading))
    .sort((a, b) => sectionBlock(a).length - sectionBlock(b).length);

  const parts: string[] = [];
  let truncated = false;

  const tryAdd = (chunk: string): boolean => {
    if (!chunk) return true;
    const next =
      parts.length === 0 ? chunk : `${parts.join("\n\n")}\n\n${chunk}`;
    if (next.length <= maxLen) {
      parts.push(chunk);
      return true;
    }
    const remaining =
      maxLen - (parts.length === 0 ? 0 : parts.join("\n\n").length + 2);
    if (remaining >= MIN_TRUNCATED_SKILL_CHARS) {
      const cut = fenceSafeTruncate(chunk, remaining);
      if (cut.length > 0) {
        parts.push(cut);
        truncated = true;
      } else {
        truncated = true;
      }
    } else {
      truncated = true;
    }
    return false;
  };

  if (intro) tryAdd(intro);
  for (const section of must) {
    if (!tryAdd(sectionBlock(section))) break;
  }
  for (const section of preferred) {
    if (!tryAdd(sectionBlock(section))) break;
  }
  for (const section of deferred) {
    if (!tryAdd(sectionBlock(section))) break;
  }

  const rebuilt = parts.join("\n\n");
  if (!rebuilt) {
    const cut = fenceSafeTruncate(trimmed, maxLen);
    return { content: cut, truncated: true };
  }
  if (rebuilt.length > maxLen) {
    return { content: fenceSafeTruncate(rebuilt, maxLen), truncated: true };
  }
  return {
    content: rebuilt,
    truncated: truncated || rebuilt.length < trimmed.length,
  };
}

function prepareSkillBlock(
  entry: CatalogSkillEntry,
  allowance: number,
): { block: string; truncated: boolean } {
  const heading = `## ${entry.name}\n`;
  const contentBudget = Math.max(0, allowance - heading.length);
  if (entry.name === AUTOMATION_SKILL_NAME) {
    const prepared = prioritizeAutomationContent(entry.content, contentBudget);
    return {
      block: `${heading}${prepared.content}`,
      truncated: prepared.truncated,
    };
  }
  const trimmed = entry.content.trim();
  const full = `${heading}${trimmed}`;
  if (full.length <= allowance) {
    return { block: full, truncated: false };
  }
  if (allowance < MIN_TRUNCATED_SKILL_CHARS) {
    return { block: "", truncated: true };
  }
  return {
    block: fenceSafeTruncate(full, allowance),
    truncated: true,
  };
}

/**
 * Pack catalog skills into the launch-addition suffix.
 *
 * Invoked skills stay first. ``openhands-automation`` (issue #16905) is packed
 * next with section-priority truncation so Managing Automations / Run Lifecycle
 * survive the 32 KiB cap. Remaining skills fill leftover budget (smallest-first).
 * Truncation is fence-safe so ``CANVAS_SKILLS_TRUNCATED`` is not swallowed by an
 * open code fence.
 */
export function packClaudeAcpSkillSuffix(
  skills: readonly CatalogSkillEntry[],
  maxLength: number = AGENT_LAUNCH_SUFFIX_APPEND_MAX_LENGTH,
  invokedName?: string,
): string | undefined {
  if (skills.length === 0) return undefined;

  const wrapPad = wrapOverhead();
  const budget = maxLength - wrapPad;
  if (budget <= 0) return undefined;

  const invoked = invokedName
    ? skills.find((entry) => entry.name === invokedName)
    : undefined;
  const primary =
    invoked?.name === AUTOMATION_SKILL_NAME
      ? undefined
      : skills.find((entry) => entry.name === AUTOMATION_SKILL_NAME);
  const rest = skills.filter(
    (entry) =>
      entry.name !== invokedName && entry.name !== AUTOMATION_SKILL_NAME,
  );
  const ordered = [
    ...(invoked ? [invoked] : []),
    ...(primary ? [primary] : []),
    ...[...rest].sort((a, b) => {
      const sizeDelta = a.content.length - b.content.length;
      if (sizeDelta !== 0) return sizeDelta;
      return a.name.localeCompare(b.name);
    }),
  ];

  const mayNeedMarker = estimatePackedBodyLength(ordered) > budget;
  const markerReserve = mayNeedMarker
    ? maxMarkerReserve(ordered.map((entry) => entry.name))
    : 0;
  const packBudget = Math.max(0, budget - markerReserve);
  if (packBudget <= 0) return undefined;

  const blocks: string[] = [];
  let used = 0;
  const truncated: string[] = [];
  const omitted: string[] = [];

  for (let i = 0; i < ordered.length; i += 1) {
    const entry = ordered[i];
    const separator = blocks.length > 0 ? "\n\n" : "";
    const available = packBudget - used - separator.length;
    if (available <= 0) {
      omitted.push(entry.name);
      continue;
    }

    const laterCount = ordered.length - i - 1;
    // Keep a modest leftover so at least one other small skill can appear when
    // packing the oversized automation skill — but never starve automation's
    // must-keep sections (those need several KB).
    const isPrimaryPack =
      entry.name === AUTOMATION_SKILL_NAME ||
      (entry.name === invokedName && entry.name === AUTOMATION_SKILL_NAME);
    const leftoverReserve =
      isPrimaryPack && laterCount > 0
        ? Math.min(Math.floor(available * 0.2), laterCount * 512)
        : 0;
    const allowance = Math.max(
      MIN_TRUNCATED_SKILL_CHARS,
      available - leftoverReserve,
    );

    const prepared = prepareSkillBlock(entry, Math.min(available, allowance));
    if (!prepared.block) {
      omitted.push(entry.name);
      continue;
    }
    if (prepared.block.length > available) {
      const tight = prepareSkillBlock(entry, available);
      if (!tight.block || tight.block.length > available) {
        omitted.push(entry.name);
        continue;
      }
      blocks.push(tight.block);
      used += separator.length + tight.block.length;
      if (tight.truncated) truncated.push(entry.name);
      continue;
    }

    blocks.push(prepared.block);
    used += separator.length + prepared.block.length;
    if (prepared.truncated) truncated.push(entry.name);
  }

  if (blocks.length === 0) return undefined;

  let body = blocks.join("\n\n");
  if (truncated.length > 0 || omitted.length > 0) {
    console.warn(
      `[acp-claude-skill-bridge] truncated or omitted Canvas skills in the ` +
        `${maxLength}-character launch suffix: ` +
        [...truncated, ...omitted].join(", "),
    );
    const marker = formatTruncationMarker({ truncated, omitted });
    const withMarker = `${body}\n\n${marker}`;
    if (wrapSkillSuffix(withMarker).length > maxLength) {
      const overflow = wrapSkillSuffix(withMarker).length - maxLength;
      body = fenceSafeTruncate(body, Math.max(0, body.length - overflow));
    }
    body = `${body}\n\n${marker}`;
    const wrapped = wrapSkillSuffix(body);
    if (wrapped.length > maxLength) {
      const bodyBudget = maxLength - wrapPad;
      body = fenceSafeTruncate(body, Math.max(0, bodyBudget));
      return wrapSkillSuffix(body);
    }
    return wrapSkillSuffix(body);
  }

  return wrapSkillSuffix(body);
}

/**
 * Build ``agent_launch_additions.system_message_suffix_append`` for a Claude
 * Code ACP launch. Returns ``undefined`` when nothing is enabled.
 */
export function buildClaudeAcpSkillSuffixAppend(options: {
  enablement: SkillEnablement;
  invokedCatalogSkill?: string;
  maxLength?: number;
}): string | undefined {
  const skills = selectClaudeAcpCatalogSkills(
    options.enablement,
    options.invokedCatalogSkill,
  );
  return packClaudeAcpSkillSuffix(
    skills,
    options.maxLength,
    options.invokedCatalogSkill,
  );
}
