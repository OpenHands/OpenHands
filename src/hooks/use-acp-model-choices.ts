import type { ACPModelOption } from "#/constants/acp-providers";
import type { MergedModelOption } from "#/api/models-dev-catalog";
import {
  useAcpCatalogModels,
  type AcpCatalogStatus,
} from "#/hooks/query/use-models-dev-catalog";
import { useAcpCustomModelsStore } from "#/stores/acp-custom-models-store";

/** Stable empty-array reference so the zustand selector never forces a
 * re-render loop when a profile has no remembered custom models. */
const EMPTY_CUSTOM_MODEL_IDS: string[] = [];

export type AcpModelChoiceSource = "curated" | "models.dev" | "custom" | "live";

/**
 * A model option offered by the ACP model picker, tagged with where it came
 * from. Shaped like {@link MergedModelOption} (id/label + optional
 * description/efforts metadata) but with a wider `source` union — a plain
 * `MergedModelOption & { source: ... }` intersection would narrow `source`
 * back down to `MergedModelOption`'s own `"curated" | "models.dev"`, so this
 * re-declares the field instead of intersecting it.
 */
export interface AcpModelChoice extends Omit<MergedModelOption, "source"> {
  source: AcpModelChoiceSource;
}

export interface BuildAcpModelChoicesInput {
  /**
   * Models the live ACP session/server currently reports (already mapped
   * from the SDK's `AcpModelInfo` `{model_id, name}` to `{id, label}` by the
   * caller — kept transport-agnostic here). Empty/omitted on surfaces with no
   * live session (e.g. the Settings → Agent profile editor).
   */
  liveModels?: ACPModelOption[];
  /** The provider's curated `available_models` list. */
  curated: ACPModelOption[];
  /** Remembered custom model ids for the profile being edited, most recently
   * added first (see `useAcpCustomModelsStore`). */
  customIds?: string[];
  /**
   * models.dev catalog models NOT already covered by `curated` (i.e. the
   * `source: "models.dev"` slice of {@link useAcpCatalogModels}'s merged
   * output — that function already dedupes catalog entries against
   * `curated`, so this only ever contains genuine extras).
   */
  catalogExtras?: MergedModelOption[];
}

/**
 * Pure merge + dedupe for the ACP model picker's option list. Order (and
 * dedupe precedence, since a later duplicate id is dropped): live session
 * models, then curated, then the profile's remembered custom entries, then
 * models.dev catalog extras. Ids are compared after trimming; the first
 * occurrence's metadata wins. Catalog extras are additionally dropped when
 * their display label matches an already-included entry — curated registry
 * ids are aliases ("sonnet") while models.dev uses full ids
 * ("claude-sonnet-4-6"), so the same model would otherwise appear twice
 * under one label. Only catalog entries get label-deduped: live, curated,
 * and user-entered custom ids are authoritative and always shown.
 */
export function buildAcpModelChoices(
  input: BuildAcpModelChoicesInput,
): AcpModelChoice[] {
  const {
    liveModels = [],
    curated,
    customIds = [],
    catalogExtras = [],
  } = input;

  const seenIds = new Set<string>();
  const seenLabels = new Set<string>();
  const choices: AcpModelChoice[] = [];

  const normalizeLabel = (label: string) => label.trim().toLowerCase();

  function pushIfNew(
    option: {
      id: string;
      label: string;
      description?: string;
      efforts?: string[];
    },
    source: AcpModelChoiceSource,
  ) {
    const id = option.id.trim();
    if (!id || seenIds.has(id)) return;
    if (source === "models.dev" && seenLabels.has(normalizeLabel(option.label)))
      return;
    seenIds.add(id);
    seenLabels.add(normalizeLabel(option.label));
    choices.push({
      id,
      label: option.label,
      source,
      description: option.description,
      efforts: option.efforts,
    });
  }

  liveModels.forEach((model) => pushIfNew(model, "live"));
  curated.forEach((model) => pushIfNew(model, "curated"));
  customIds.forEach((id) => pushIfNew({ id, label: id }, "custom"));
  catalogExtras.forEach((model) => pushIfNew(model, "models.dev"));

  return choices;
}

export interface UseAcpModelChoicesInput {
  /** ACP registry key (or `ACP_CUSTOM_PRESET_KEY`) the picker is showing
   * options for; `null`/`undefined` when no provider/command is selected
   * yet. Forwarded to {@link useAcpCatalogModels} to resolve the models.dev
   * provider mapping. */
  acpServer: string | null | undefined;
  /** The provider's curated `available_models` list (`[]` for a custom ACP
   * server, which has no curated list of its own). */
  curated: ACPModelOption[];
  /** Stable AgentProfile UUID being edited, when known. `undefined` in the
   * profile-creation flow (no id minted yet) — in that case no custom
   * entries are offered and none can be remembered, matching pre-M2
   * behavior for new profiles. */
  profileId?: string;
  /** See {@link BuildAcpModelChoicesInput.liveModels}. */
  liveModels?: ACPModelOption[];
  /**
   * Forwarded to {@link useAcpCatalogModels} — pass `false` to skip the
   * models.dev catalog fetch when this hook must run unconditionally (Rules
   * of Hooks) but the picker won't be shown this render (e.g. a non-ACP
   * chat). Defaults to `true`, matching every pre-M3 call site.
   */
  enabled?: boolean;
}

export interface UseAcpModelChoicesResult {
  choices: AcpModelChoice[];
  catalogStatus: AcpCatalogStatus;
}

/**
 * Shared selector behind the ACP model picker: curated models, the profile's
 * remembered custom entries, and models.dev catalog extras, merged into one
 * deduped list (see {@link buildAcpModelChoices} for order/precedence).
 * Reused as-is by the Settings → Agent profile editor (no `liveModels`) and,
 * from M3, by a live-session picker that does pass them.
 */
export function useAcpModelChoices({
  acpServer,
  curated,
  profileId,
  liveModels,
  enabled,
}: UseAcpModelChoicesInput): UseAcpModelChoicesResult {
  const { models: catalogMerged, catalogStatus } = useAcpCatalogModels(
    acpServer,
    curated,
    { enabled },
  );
  const catalogExtras = catalogMerged.filter(
    (model) => model.source === "models.dev",
  );

  const customIds = useAcpCustomModelsStore((state) =>
    profileId
      ? (state.customModelsByProfileId[profileId] ?? EMPTY_CUSTOM_MODEL_IDS)
      : EMPTY_CUSTOM_MODEL_IDS,
  );

  const choices = buildAcpModelChoices({
    liveModels,
    curated,
    customIds,
    catalogExtras,
  });

  return { choices, catalogStatus };
}
