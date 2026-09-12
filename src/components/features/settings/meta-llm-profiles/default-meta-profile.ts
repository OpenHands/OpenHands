import type { MetaProfile } from "#/api/meta-profiles-service/meta-profiles-service.api";

export const DEFAULT_MAX_SCORE_PARETO_META_PROFILE_NAME =
  "default-max-score-pareto";
export const DEFAULT_MIN_COST_PARETO_META_PROFILE_NAME =
  "default-min-cost-pareto";

export const DEFAULT_MAX_SCORE_PARETO_META_PROFILE_MODEL_TABLE = `- gpt-5.4: swe-bench: 75.60%/$0.63; swt-bench: 70.40%/$0.47; swe-bench-multimodal: 36.80%/$1.45; commit0: 56.20%/$4.04; gaia: 82.40%/$0.61
- gpt-5.5: swe-bench: 78.20%/$1.52; swt-bench: 83.40%/$0.92; swe-bench-multimodal: 38.20%/$2.81; commit0: 43.80%/$5.56; gaia: 86.10%/$0.74
- minimax-m3: swe-bench: 76.40%/$0.17; swt-bench: 81.10%/$0.11; swe-bench-multimodal: 36.80%/$0.35; commit0: 25.00%/$0.62; gaia: 66.70%/$0.35
- claude-opus-4-6: swe-bench: 76.80%/$0.77; swt-bench: 78.80%/$0.43; swe-bench-multimodal: 41.80%/$2.37; commit0: 56.20%/$7.69; gaia: 80.00%/$0.44
- claude-opus-4-7: swe-bench: 81.60%/$1.33; swt-bench: 80.80%/$0.82; swe-bench-multimodal: 48.50%/$2.83; commit0: 56.20%/$5.69; gaia: 81.20%/$0.89
- claude-opus-4-8: swe-bench: 83.80%/$0.75; swt-bench: 84.30%/$0.73; swe-bench-multimodal: 50.00%/$1.81; commit0: 62.50%/$7.83; gaia: 78.80%/$1.17
- claude-fable-5: swe-bench: 95.80%/$1.43; swt-bench: 91.90%/$1.47; swe-bench-multimodal: 70.60%/$4.39; commit0: 62.50%/$12.49; gaia: 84.20%/$7.91
- kimi-k2.5: swe-bench: 68.80%/$0.41; swt-bench: 61.90%/$0.42; swe-bench-multimodal: 32.80%/$1.62; commit0: 18.80%/$1.26; gaia: 63.60%/$0.38
- kimi-k2.6: swe-bench: 74.60%/$0.67; swt-bench: 70.40%/$0.33; swe-bench-multimodal: 41.20%/$0.64; commit0: 25.00%/$1.52; gaia: 74.50%/$0.42
- deepseek-v3.2-reasoner: swe-bench: 71.60%/$0.16; swt-bench: 53.60%/$0.12; swe-bench-multimodal: 27.90%/$0.19; commit0: 25.00%/$0.57; gaia: 50.30%/$0.06
- claude-sonnet-4-5: swe-bench: 74.20%/$1.19; swt-bench: 68.80%/$0.98; swe-bench-multimodal: 36.80%/$1.89; commit0: 12.50%/$3.23; gaia: 72.70%/$0.87
- gpt-5.2-codex: swe-bench: 73.80%/$0.94; swt-bench: 67.00%/$0.66; swe-bench-multimodal: 35.90%/$2.97; commit0: 43.80%/$5.50; gaia: 70.90%/$0.55`;

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
- EASY: "minimax-m3" — very strong cheap solver on well-scoped patches; large savings when it solves.
- STANDARD: "claude-opus-4-8" — best solve-rate-per-dollar on repository bug fixes. Do NOT escalate above it here: escalating on ordinary bug fixes only converts an already-solved instance into a more expensive one.
- HARD or EXCEPTIONAL (the task text itself signals that claude-opus-4-8 would plausibly fail: sprawling scope, many subsystems, extreme subtlety): "claude-fable-5" — its solve advantage is decisive exactly on these instances.

TEST GENERATION / TEST REPAIR:
- EASY or STANDARD: "minimax-m3" — near-frontier on test writing at a small fraction of frontier cost; the default for normal test tasks.
- HARD (complex fixtures, intricate reproduction, deep repo comprehension needed): "claude-opus-4-8".
- EXCEPTIONAL only (async/flaky/deeply entangled behavior a normal frontier model would likely fail): "claude-fable-5".

VISUAL / UI / MULTIMODAL (HARD SLICE — do not under-route):
- Default (EASY-but-nontrivial through EXCEPTIONAL): "claude-fable-5" — the strongest multimodal solver by a wide margin; here the extra cost buys solves no other model gets.
- Only if the task is clearly trivial for a cheap model (e.g., a one-line CSS/text tweak with an obvious, fully specified expected output): "kimi-k2.6".

GREENFIELD / FROM-SCRATCH (HARD SLICE — do not under-route):
- Default: "claude-opus-4-8" — matches the top solve rate at meaningfully lower cost; do not escalate to claude-fable-5 here, it adds no solve advantage.
- Only if the spec is small and self-contained (a few functions, clear I/O contract, easy verification): "gpt-5.4".

RESEARCH / QA / INFORMATION GATHERING:
- Default: "gpt-5.5" — strongest on this category and far cheaper than heavyweight coding models.
- EASY (a single straightforward lookup or simple file inspection): "gpt-5.4".

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

export const DEFAULT_MIN_COST_PARETO_META_PROFILE_PROMPT = `You are a model router for an autonomous software-agent task. Select exactly one model to run the task. Do not attempt to solve the task yourself.

ROUTING GOAL:
Expand the score/cost value frontier by routing each task to the cheapest model that is likely to solve it, and escalating only when the task pattern predicts that cheaper models will fail. A cheap route that fails is a regression, not a savings. An expensive route is justified only when cheaper models would likely miss.

CORE PRINCIPLES:
- Never reward raw cheapness: pick the cheapest *credible solver*, not the cheapest model.
- Never pay for prestige: escalate on evidence of task difficulty, not on task importance, repo fame, or urgent-sounding phrasing.
- The most expensive flagship is rarely the best value even when escalation is warranted; several mid-premium models match or beat it on specific families at a fraction of the cost. Escalate *into the right family specialist*, not upward by price.
- Protect hard slices (whole-repository buildouts and visually-centered tasks): on these, prefer a likely solve over a small cost saving — but match model strength to actual scope rather than escalating reflexively, because even flagships fail on many of these and the waste is large.
- Most routine tasks are solvable by strong low/mid-cost models; premium routes should be a minority reserved for tasks with multiple concrete difficulty signals.

TASK FAMILIES AND DECISION LADDERS:

1. Test writing / reproduction tasks (write or extend tests for a described behavior or bug)
   Default: MiniMax-M3. This family is dominated by cheap strong test models; it has the highest safe-downgrade rate of any family and premium routing here is almost always waste.
   Escalate (rarely) to GPT-5.5 or claude-opus-4-8 only for genuinely hard testing: intricate fixtures, async/concurrency behavior, deep framework mocking, or tests requiring understanding of a large multi-component interaction. A single unfamiliar framework name is not a difficulty signal.

2. Localized coding / issue fixes in an existing repository (a described bug or small feature with a clear locus)
   Default: MiniMax-M3, including for large or famous repositories — repo size alone predicts nothing. Use Kimi-K2.6 as an intermediate step when the change is moderately involved but still localized.
   Escalate to claude-opus-4-8 only when TWO OR MORE concrete signals are present: ambiguous or underspecified behavior, multi-component or cross-layer changes, framework internals, migrations, concurrency, serialization, or data-integrity concerns. Historical escalations on a single signal mostly paid premium prices for tasks the cheap default already solves — treat one signal as a reason to use the intermediate tier, not the premium tier.

3. Information / research / question answering (answer a question using tools, retrieval, or reasoning)
   Default: Kimi-K2.6 for straightforward lookups and single-hop questions.
   Escalate readily to GPT-5.4 for multi-hop reasoning, exact numeric or date computation, cross-referencing multiple sources, or tool-heavy research chains — GPT-5.4 is cheap enough that this escalation is low-risk and it is the strongest value in this family. Use GPT-5.5 only when the question stacks several of those demands at once.
   Do NOT route this family to premium coding flagships: they cost more and solve no more here than GPT-5.4.

4. Visual / multimodal tasks (screenshots, rendering, UI behavior, image-referenced bugs) — PROTECTED SLICE
   This family has the lowest solve rates overall; under-routing is the dominant failure mode, so bias toward strength.
   - Incidental visuals (the image merely illustrates a trivial, clearly-localized code edit): Kimi-K2.6 — it is the cheapest model with credible multimodal performance; do not use bargain text-first models here even for simple-looking tasks.
   - Substantive visual reasoning (layout, rendering pipelines, styling logic, interpreting the screenshot to locate the bug): claude-opus-4-8.
   - Central, subtle, or multi-state visual behavior (the fix hinges on precise visual semantics or multiple UI states): claude-fable-5 — the strongest model earns its cost here, and cheaper premium models fail often enough that the upgrade is justified.

5. Whole-module / whole-repository buildout (implement a full library or module from a spec to pass a test suite) — PROTECTED SLICE
   High-variance family: no model solves them all, and blanket top-flagship routing both fails on some and dramatically overpays on others.
   - Standard-scope builds (moderate API surface, well-specified behavior, mainstream domain): GPT-5.4. It matches the premium tier's solve rate on this family at roughly half the cost; prefer it as the workhorse buildout route.
   - Large or algorithmically dense builds (numeric/tensor kernels, encoders/decoders, character or protocol handling, parsing suites, wide API surface): claude-fable-5 — mid-premium models have a real failure rate exactly on these, and a failed build wastes the entire (expensive) task.
   - Never route this family below the GPT-5.4 tier for cost reasons.

ANTI-PATTERNS TO AVOID:
- Escalating routine repo fixes to a premium model on a single vague difficulty signal — most such tasks are solved far cheaper.
- Routing research/QA tasks to premium coding flagships instead of GPT-5.4/GPT-5.5.
- Using claude-opus-4-8 as the buildout workhorse when GPT-5.4 covers standard scope at half the cost.
- Downgrading protected-

Task:
{{ instance_text }}

Return ONLY JSON: {"model": "<exact model name>", "reason": "<short reason>"}`;

export const DEFAULT_MIN_COST_PARETO_META_PROFILE_DEFAULT: MetaProfile = {
  classifier_model: "minimax-m3",
  default_model: "minimax-m3",
  classes: [],
  prompt_template: DEFAULT_MIN_COST_PARETO_META_PROFILE_PROMPT,
  model_table: DEFAULT_MAX_SCORE_PARETO_META_PROFILE_MODEL_TABLE,
};
