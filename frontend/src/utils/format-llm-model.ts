function capitalize(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : s;
}

// Reasoning-effort qualifiers used as the final slash-segment in model names
// (e.g. "gpt-5.5/high", "o3/medium", "gpt-5.5/xhigh"). Keeping this list short avoids
// false-positive matches on real model name segments.
const EFFORT_QUALS = new Set(["high", "medium", "low", "xhigh", "none"]);

/**
 * Format a raw ``llm_model`` routing string into a short, human-readable
 * label suitable for the conversation chip.
 *
 * Examples:
 *   "anthropic/claude-sonnet-4-5-20250929" -> "Claude Sonnet 4.5"
 *   "openai/gpt-4o"                        -> "GPT-4o"
 *   "openai/gpt-4o-mini"                   -> "GPT-4o mini"
 *   "openai/gpt-5.5/high"                  -> "GPT-5.5 (high)"
 *   "gemini/gemini-2.5-pro"                -> "Gemini 2.5 Pro"
 *   "openhands/o3"                         -> "o3"
 *   "openai/o4-mini/medium"                -> "o4-mini (medium)"
 *   "litellm_proxy/anthropic/claude-3-5-sonnet-20241022" -> "Claude 3.5 Sonnet"
 *   "litellm_proxy/my-finetune"            -> "my-finetune"
 *
 * Unknown models pass through with the routing prefix stripped. The raw
 * string should still be surfaced as a tooltip elsewhere.
 */
export function formatLlmModel(raw: string): string {
  if (!raw) return raw;

  // Strip routing prefix, preserving model-internal slashes when the last
  // segment is a known effort qualifier (e.g. "openai/gpt-5.5/high" → "gpt-5.5/high").
  const lastSlash = raw.lastIndexOf("/");
  const lastSegment = lastSlash >= 0 ? raw.slice(lastSlash + 1) : raw;
  let stripped: string;
  if (EFFORT_QUALS.has(lastSegment.toLowerCase()) && lastSlash >= 0) {
    // Back up one level so the effort qualifier stays attached to the model.
    const prevSlash = raw.lastIndexOf("/", lastSlash - 1);
    stripped = prevSlash >= 0 ? raw.slice(prevSlash + 1) : raw;
  } else {
    stripped = lastSegment;
  }

  // Strip trailing date suffix: -YYYYMMDD or -YYYY-MM-DD.
  const noDate = stripped
    .replace(/-\d{8}$/, "")
    .replace(/-\d{4}-\d{2}-\d{2}$/, "");

  const lower = noDate.toLowerCase();

  // Claude (4.x+ naming): claude-{tier}-{maj}[-{min}]
  //   claude-sonnet-4-5 -> Claude Sonnet 4.5
  //   claude-opus-4     -> Claude Opus 4
  let m = lower.match(/^claude-(sonnet|opus|haiku)-(\d+)(?:-(\d+))?$/);
  if (m) {
    const tier = capitalize(m[1]);
    const version = m[3] ? `${m[2]}.${m[3]}` : m[2];
    return `Claude ${tier} ${version}`;
  }

  // Claude (3.x naming): claude-{maj}[-{min}]-{tier}
  //   claude-3-5-sonnet -> Claude 3.5 Sonnet
  //   claude-3-opus     -> Claude 3 Opus
  m = lower.match(/^claude-(\d+)(?:-(\d+))?-(sonnet|opus|haiku)$/);
  if (m) {
    const version = m[2] ? `${m[1]}.${m[2]}` : m[1];
    const tier = capitalize(m[3]);
    return `Claude ${version} ${tier}`;
  }

  // Claude bare aliases: opus[1m], sonnet[1m], opusplan, etc.
  //   opus[1m]     -> Claude Opus (1M)
  //   sonnet[1m]   -> Claude Sonnet (1M)
  //   opusplan     -> Claude Opus (plan)
  m = lower.match(/^(opus|sonnet|haiku)(?:\[(.+)\])?(?:-(.+))?$/);
  if (m) {
    const tier = capitalize(m[1]);
    let qualifier = "";
    if (m[2]) {
      qualifier = m[2].toUpperCase();
    } else if (m[3]) {
      qualifier = capitalize(m[3]);
    }
    return qualifier ? `Claude ${tier} (${qualifier})` : `Claude ${tier}`;
  }

  // GPT family: gpt-{rest}
  //   gpt-4o           -> GPT-4o
  //   gpt-4o-mini      -> GPT-4o mini
  //   gpt-4.1          -> GPT-4.1
  //   gpt-5            -> GPT-5
  //   gpt-5.5/high     -> GPT-5.5 (high)
  //   gpt-5.3-codex    -> GPT-5.3 Codex
  //   gpt-5.3-codex/high -> GPT-5.3 Codex (high)
  m = lower.match(/^gpt-(.+)$/);
  if (m) {
    const rest = m[1];
    // Reasoning-effort qualifier via slash: "5.5/high" → "GPT-5.5 (high)"
    const slashIdx = rest.indexOf("/");
    let beforeSlash = rest;
    let effort = "";
    if (slashIdx >= 0) {
      beforeSlash = rest.slice(0, slashIdx);
      effort = rest.slice(slashIdx + 1);
    }
    // Check for codex suffix
    if (beforeSlash.endsWith("-codex")) {
      const versionMatch = beforeSlash.slice(0, -6).match(/^[\d.]+/);
      if (versionMatch) {
        const version = versionMatch[0];
        return effort
          ? `GPT-${version} Codex (${effort})`
          : `GPT-${version} Codex`;
      }
    }
    // Regular GPT suffix
    const versionMatch = beforeSlash.match(/^([\d.]+[a-z]?)(?:-(.+))?$/);
    if (versionMatch) {
      const version = versionMatch[1];
      const suffix = versionMatch[2];
      const text = suffix ? `GPT-${version} ${suffix}` : `GPT-${version}`;
      return effort ? `${text} (${effort})` : text;
    }
    return effort ? `GPT-${beforeSlash} (${effort})` : `GPT-${beforeSlash}`;
  }

  // o-series (OpenAI reasoning): o1, o3, o3-mini, o4-mini, o3/high, etc.
  //   o3        -> o3
  //   o4-mini   -> o4-mini
  //   o3/high   -> o3 (high)
  m = lower.match(/^(o\d+(?:-[a-z]+)?)(?:\/(.+))?$/);
  if (m) {
    return m[2] ? `${m[1]} (${m[2]})` : m[1];
  }

  // Gemini: gemini-{ver}[-{tier}[-{rest}]]
  //   gemini-2.5-pro       -> Gemini 2.5 Pro
  //   gemini-2.0-flash-001 -> Gemini 2.0 Flash 001
  m = lower.match(/^gemini-(\d+(?:\.\d+)?)(?:-(.+))?$/);
  if (m) {
    const ver = m[1];
    const tail = m[2];
    if (!tail) return `Gemini ${ver}`;
    const tailWords = tail.split("-").map(capitalize).join(" ");
    return `Gemini ${ver} ${tailWords}`;
  }

  // Unknown — return the prefix-stripped form so the chip stays readable.
  return noDate;
}
