/**
 * Actionable hints for known service start-up failures.
 *
 * A service that dies during start-up currently surfaces as a wall of build
 * output followed by `Exited with code 1`, and the launcher carries on. The
 * user is left to work out from a Rust backtrace that their toolchain is too
 * old. These matchers turn the handful of failures we can recognise into a
 * one-line explanation plus the command that fixes it.
 *
 * Kept as pure functions over already-captured output so they can be unit
 * tested without spawning anything.
 */

/** How many recent output lines a service keeps for post-mortem matching. */
export const SERVICE_OUTPUT_BUFFER_LINES = 200;

// `error: rustc 1.93.1 is not supported by the following packages:`
const RUSTC_UNSUPPORTED_RE = /rustc\s+(\d+\.\d+(?:\.\d+)?)\s+is not supported/i;
// `aws-config@1.9.0 requires rustc 1.94.1`
const RUSTC_REQUIRED_RE = /requires rustc\s+(\d+\.\d+(?:\.\d+)?)/i;
// uv/maturin wrapper lines seen when a native wheel fails to build.
const MATURIN_FAILURE_RE =
  /maturin\.build_wheel.*failed|Failed to build `[^`]+`/i;

/**
 * Highest version string in `versions`, compared numerically per component.
 * The AWS crates in the litellm tree all name the same requirement today, but
 * nothing guarantees that, so take the strictest rather than the first seen.
 */
function highestVersion(versions) {
  const parse = (v) => v.split(".").map((n) => Number.parseInt(n, 10) || 0);
  return versions.reduce((best, v) => {
    if (!best) return v;
    const [a, b] = [parse(v), parse(best)];
    for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
      if ((a[i] ?? 0) > (b[i] ?? 0)) return v;
      if ((a[i] ?? 0) < (b[i] ?? 0)) return best;
    }
    return best;
  }, null);
}

/**
 * Recognise a Rust toolchain that is too old to build a native wheel.
 *
 * Returns null unless the output both fails a wheel build and names an
 * unsupported rustc — a bare `Failed to build` can be any number of things,
 * and guessing wrong would be worse than staying quiet.
 */
export function detectToolchainFailure(lines) {
  const text = Array.isArray(lines) ? lines.join("\n") : String(lines ?? "");
  const unsupported = text.match(RUSTC_UNSUPPORTED_RE);
  if (!unsupported || !MATURIN_FAILURE_RE.test(text)) return null;

  const required = highestVersion(
    [...text.matchAll(new RegExp(RUSTC_REQUIRED_RE, "gi"))].map((m) => m[1]),
  );

  return {
    kind: "rust-toolchain-too-old",
    found: unsupported[1],
    required,
    summary: required
      ? `Rust toolchain too old: found rustc ${unsupported[1]}, need ${required} or newer.`
      : `Rust toolchain too old: found rustc ${unsupported[1]}.`,
    // Both remediations are the ones that actually worked for reporters on
    // the issue: rustup for rustup-managed installs, brew for Homebrew ones.
    remedies: [
      "rustup update stable && rustup default stable",
      "brew upgrade rust   # if rustc came from Homebrew",
    ],
  };
}

/** All matchers, in priority order. */
const DETECTORS = [detectToolchainFailure];

/**
 * Explain why `name` exited, given the output it produced.
 * Returns null when the failure is not one we recognise, so callers keep the
 * existing generic message rather than inventing an explanation.
 */
export function describeServiceFailure(name, exitCode, lines) {
  for (const detect of DETECTORS) {
    const hit = detect(lines);
    if (!hit) continue;
    return {
      ...hit,
      service: name,
      exitCode,
      lines: [
        `${name} could not start: ${hit.summary}`,
        ...hit.remedies.map((r) => `  ${r}`),
        "  Then restart agent-canvas.",
      ],
    };
  }
  return null;
}
