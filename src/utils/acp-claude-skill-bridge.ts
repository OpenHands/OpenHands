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

/**
 * Pack catalog skills into the launch-addition suffix.
 *
 * Invoked skills stay first. Remaining skills are packed largest-first so a
 * single oversized skill such as ``openhands-automation`` (~39 KiB) is not
 * crowded out by smaller default-enabled catalog entries. Content that still
 * cannot fit is truncated rather than dropped, because omitting the skill
 * entirely reproduces #16905.
 */
export function packClaudeAcpSkillSuffix(
  skills: readonly CatalogSkillEntry[],
  maxLength: number = AGENT_LAUNCH_SUFFIX_APPEND_MAX_LENGTH,
  invokedName?: string,
): string | undefined {
  if (skills.length === 0) return undefined;

  const budget = maxLength - wrapOverhead();
  if (budget <= 0) return undefined;

  const invoked = invokedName
    ? skills.find((entry) => entry.name === invokedName)
    : undefined;
  const rest = skills.filter((entry) => entry.name !== invokedName);
  const ordered = [
    ...(invoked ? [invoked] : []),
    ...[...rest].sort((a, b) => {
      const sizeDelta = b.content.length - a.content.length;
      if (sizeDelta !== 0) return sizeDelta;
      return a.name.localeCompare(b.name);
    }),
  ];

  const blocks: string[] = [];
  let used = 0;
  const dropped: string[] = [];

  for (const entry of ordered) {
    const block = formatSkillBlock(entry);
    const separator = blocks.length > 0 ? "\n\n" : "";
    const piece = `${separator}${block}`;
    if (used + piece.length <= budget) {
      blocks.push(block);
      used += piece.length;
      continue;
    }
    const remaining = budget - used - separator.length;
    if (remaining >= MIN_TRUNCATED_SKILL_CHARS) {
      blocks.push(block.slice(0, remaining));
      used = budget;
      dropped.push(entry.name);
      // Budget exhausted by the truncated skill.
      const remainingNames = ordered
        .slice(ordered.indexOf(entry) + 1)
        .map((skill) => skill.name);
      dropped.push(...remainingNames);
      break;
    }
    dropped.push(entry.name);
  }

  if (blocks.length === 0) return undefined;
  if (dropped.length > 0) {
    console.warn(
      `[acp-claude-skill-bridge] truncated or omitted Canvas skills in the ` +
        `${maxLength}-character launch suffix: ${dropped.join(", ")}`,
    );
  }
  return wrapSkillSuffix(blocks.join("\n\n"));
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
