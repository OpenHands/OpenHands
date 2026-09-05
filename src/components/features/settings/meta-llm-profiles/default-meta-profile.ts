import type { MetaProfile } from "#/api/meta-profiles-service/meta-profiles-service.api";

export const DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME =
  "default-max-score-pareto";
export const DEFAULT_MIN_COST_PARETO_META_PROFILE_NAME =
  "default-min-cost-pareto";

export const DEFAULT_MAX_SCORE_PARETO_META_PROFILE_MODEL_TABLE = `- GPT-5.4: swe-bench: 75.60%/$0.63; swt-bench: 70.40%/$0.47; swe-bench-multimodal: 36.80%/$1.45; commit0: 56.20%/$4.04; gaia: 82.40%/$0.61
- GPT-5.5: swe-bench: 78.20%/$1.52; swt-bench: 83.40%/$0.92; swe-bench-multimodal: 38.20%/$2.81; commit0: 43.80%/$5.56; gaia: 86.10%/$0.74
- MiniMax-M3: swe-bench: 76.40%/$0.17; swt-bench: 81.10%/$0.11; swe-bench-multimodal: 36.80%/$0.35; commit0: 25.00%/$0.62; gaia: 66.70%/$0.35
- claude-opus-4-6: swe-bench: 76.80%/$0.77; swt-bench: 78.80%/$0.43; swe-bench-multimodal: 41.80%/$2.37; commit0: 56.20%/$7.69; gaia: 80.00%/$0.44
- claude-opus-4-7: swe-bench: 81.60%/$1.33; swt-bench: 80.80%/$0.82; swe-bench-multimodal: 48.50%/$2.83; commit0: 56.20%/$5.69; gaia: 81.20%/$0.89
- claude-opus-4-8: swe-bench: 83.80%/$0.75; swt-bench: 84.30%/$0.73; swe-bench-multimodal: 50.00%/$1.81; commit0: 62.50%/$7.83; gaia: 78.80%/$1.17
- claude-fable-5: swe-bench: 95.80%/$1.43; swt-bench: 91.90%/$1.47; swe-bench-multimodal: 70.60%/$4.39; commit0: 62.50%/$12.49; gaia: 84.20%/$7.91
- Kimi-K2.5: swe-bench: 68.80%/$0.41; swt-bench: 61.90%/$0.42; swe-bench-multimodal: 32.80%/$1.62; commit0: 18.80%/$1.26; gaia: 63.60%/$0.38
- Kimi-K2.6: swe-bench: 74.60%/$0.67; swt-bench: 70.40%/$0.33; swe-bench-multimodal: 41.20%/$0.64; commit0: 25.00%/$1.52; gaia: 74.50%/$0.42
- DeepSeek-V3.2-Reasoner: swe-bench: 71.60%/$0.16; swt-bench: 53.60%/$0.12; swe-bench-multimodal: 27.90%/$0.19; commit0: 25.00%/$0.57; gaia: 50.30%/$0.06
- claude-sonnet-4-5: swe-bench: 74.20%/$1.19; swt-bench: 68.80%/$0.98; swe-bench-multimodal: 36.80%/$1.89; commit0: 12.50%/$3.23; gaia: 72.70%/$0.87
- GPT-5.2-Codex: swe-bench: 73.80%/$0.94; swt-bench: 67.00%/$0.66; swe-bench-multimodal: 35.90%/$2.97; commit0: 43.80%/$5.50; gaia: 70.90%/$0.55`;

export const DEFAULT_MAX_SCORE_PARETO_META_PROFILE_PROMPT = `You are a model router for an autonomous software agent. Your job is to pick exactly one model for the task below. Do not solve the task; never answer it.

Step 1 — Classify the task into exactly one category:
- BUG-FIX / CODE REPAIR: fix a defect or implement a change in an existing, mature repository (SWE-bench-like).
- TEST GENERATION / TEST REPAIR: write or fix tests that reproduce or verify behavior; the deliverable is tests, not the fix (SWT-bench-like).
- VISUAL / UI / MULTIMODAL: task references screenshots, rendered output, CSS/SVG/canvas, charts, images, or frontend visual behavior (SWE-bench-multimodal-like). HARD SLICE.
- GREENFIELD / FROM-SCRATCH IMPLEMENTATION: build a library or module from a spec with sparse or skeleton existing code; many functions to implement against a test suite (Commit0-like). HARD SLICE.
- RESEARCH / QA / INFORMATION GATHERING: web lookup, multi-hop factual reasoning, file/data inspection questions (GAIA-like).

Step 2 — Judge difficulty within the category using four tiers:
- EASY: small, localized change; clear reproduction; single file or function; unambiguous spec; simple lookup. A strong cheap model very likely solves it.
- STANDARD: a typical instance — nontrivial but bounded (a focused patch, a coherent test-suite change, a multi-step but contained question).
- HARD: cross-cutting or subtle; multiple interacting components, tricky reproduction, ambiguous spec, or deep domain knowledge — the kind of instance a strong frontier model plausibly fails.
- EXCEPTIONAL: clearly among the hardest instances — sprawling changes across many subsystems, deeply ambiguous or contradictory requirements, tasks that even top frontier models routinely fail.

Step 3 — Route using these rules (exact model names):

BUG-FIX / CODE REPAIR:
- EASY: "MiniMax-M3" — very strong cheap solver on well-scoped patches; large savings when it solves.
- STANDARD: "claude-opus-4-8" — best solve-rate-per-dollar on repository bug fixes. Do NOT escalate above it here: escalating on ordinary bug fixes only converts an already-solved instance into a more expensive one.
- HARD or EXCEPTIONAL (the task text itself signals that claude-opus-4-8 would plausibly fail: sprawling scope, many subsystems, extreme subtlety): "claude-fable-5" — its solve advantage is decisive exactly on these instances.

TEST GENERATION / TEST REPAIR:
- EASY or STANDARD: "MiniMax-M3" — near-frontier on test writing at a small fraction of frontier cost; the default for normal test tasks.
- HARD (complex fixtures, intricate reproduction, deep repo comprehension needed): "claude-opus-4-8".
- EXCEPTIONAL only (async/flaky/deeply entangled behavior a normal frontier model would likely fail): "claude-fable-5".

VISUAL / UI / MULTIMODAL (HARD SLICE — do not under-route):
- Default (EASY-but-nontrivial through EXCEPTIONAL): "claude-fable-5" — the strongest multimodal solver by a wide margin; here the extra cost buys solves no other model gets.
- Only if the task is clearly trivial for a cheap model (e.g., a one-line CSS/text tweak with an obvious, fully specified expected output): "Kimi-K2.6".

GREENFIELD / FROM-SCRATCH (HARD SLICE — do not under-route):
- Default: "claude-opus-4-8" — matches the top solve rate at meaningfully lower cost; do not escalate to claude-fable-5 here, it adds no solve advantage.
- Only if the spec is small and self-contained (a few functions, clear I/O contract, easy verification): "GPT-5.4".

RESEARCH / QA / INFORMATION GATHERING:
- Default: "GPT-5.5" — strongest on this category and far cheaper than heavyweight coding models.
- EASY (a single straightforward lookup or simple file inspection): "GPT-5.4".

Tie-breaking principles:
- A cheap model only helps if it actually solves; a cheap failure saves nothing and loses the instance. Never pick a cheap model on a HARD SLICE or a HARD/EXCEPTIONAL instance just to reduce cost.
- Equally, never pay a premium for capability that adds nothing: if claude-opus-4-8 is likely to solve, do not escalate to claude-fable-5. Escalate ONLY where its solve advantage is decisive: multimodal tasks, or code/test tasks that claude-opus-4-8 would likely fail.
- Prefer routes likely to solve instances that claude-opus-4-8 would fail; keep claude-opus-4-8 on hard cases it solves.
- If two models are similarly likely to solve, choose the cheaper one.
- If the category is genuinely ambiguous, treat it as BUG-FIX / CODE REPAIR and route by difficulty.
- When difficulty is uncertain between two tiers on a HARD SLICE, choose the higher tier; elsewhere, choose the tier the evidence in the task text best supports.

You may choose any model listed in the model table below; the recommendations above should be followed unless the task text gives a strong, specific reason to deviate.

{{ model_table }}

Return ONLY valid JSON in this exact shape, with the model field containing an exact model name from the table:
{"model": "<exact model name>", "reason": "<short reason: category + difficulty tier + why this model>"}

Task:
{{ instance_text }}`;

export const DEFAULT_MAX_SCORE_PARETO_META_PROFILE_DEFAULT: MetaProfile = {
  classifier_model: "minimax-m3",
  default_model: "minimax-m3",
  classes: [],
  prompt_template: DEFAULT_MAX_SCORE_PARETO_META_PROFILE_PROMPT,
  model_table: DEFAULT_MAX_SCORE_PARETO_META_PROFILE_MODEL_TABLE,
};

export const DEFAULT_MIN_COST_PARETO_META_PROFILE_PROMPT = `Placeholder min-cost Pareto router prompt.

Use the model table below to choose the lowest-cost model that is likely to solve the task. Return only valid JSON with the exact model name.

{{ model_table }}

Task:
{{ instance_text }}`;

export const DEFAULT_MIN_COST_PARETO_META_PROFILE_DEFAULT: MetaProfile = {
  classifier_model: "minimax-m3",
  default_model: "minimax-m3",
  classes: [],
  prompt_template: DEFAULT_MIN_COST_PARETO_META_PROFILE_PROMPT,
  model_table: DEFAULT_MAX_SCORE_PARETO_META_PROFILE_MODEL_TABLE,
};
