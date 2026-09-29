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

/**
 * Pack catalog skills into the launch-addition suffix.
 *
 * Invoked skills stay first. Remaining skills are ordered smallest-first and
 * packed under a reserved per-skill share of the 32 KiB budget so several
 * enabled catalog skills survive instead of one oversized skill silently
 * consuming the whole cap. Content that still cannot fit its share is
 * truncated (never dropped below ``MIN_TRUNCATED_SKILL_CHARS`` when the share
 * allows); when anything is truncated or omitted, a machine-readable
 * ``CANVAS_SKILLS_TRUNCATED`` marker listing those names is appended.
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
  const rest = skills.filter((entry) => entry.name !== invokedName);
  const ordered = [
    ...(invoked ? [invoked] : []),
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
    const block = formatSkillBlock(entry);
    const separator = blocks.length > 0 ? "\n\n" : "";
    const remainingSkills = ordered.length - i;
    const available = packBudget - used - separator.length;
    if (available <= 0) {
      omitted.push(entry.name);
      continue;
    }
    // Reserved fair share of what is left for this and later skills.
    const share = Math.floor(available / remainingSkills);
    const allowance = Math.max(share, 0);

    if (block.length <= available && block.length <= allowance) {
      blocks.push(block);
      used += separator.length + block.length;
      continue;
    }

    // Prefer a full skill when it fits the remaining budget even if above the
    // equal share — leftover from smaller earlier skills funds this. Only do
    // so for the last remaining skill, or when every later skill can still get
    // MIN_TRUNCATED_SKILL_CHARS after taking the full block.
    const laterCount = remainingSkills - 1;
    const afterFull = available - block.length;
    const canTakeFull =
      block.length <= available &&
      (laterCount === 0 || afterFull >= laterCount * MIN_TRUNCATED_SKILL_CHARS);

    if (canTakeFull) {
      blocks.push(block);
      used += separator.length + block.length;
      continue;
    }

    const truncateTo = Math.min(available, Math.max(allowance, share));
    if (truncateTo >= MIN_TRUNCATED_SKILL_CHARS) {
      blocks.push(block.slice(0, truncateTo));
      used += separator.length + truncateTo;
      truncated.push(entry.name);
      continue;
    }

    omitted.push(entry.name);
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
      body = body.slice(0, Math.max(0, body.length - overflow));
    }
    body = `${body}\n\n${marker}`;
    // Final hard clamp — should be a no-op when reserve was accurate.
    const wrapped = wrapSkillSuffix(body);
    if (wrapped.length > maxLength) {
      const bodyBudget = maxLength - wrapPad;
      body = body.slice(0, Math.max(0, bodyBudget));
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
