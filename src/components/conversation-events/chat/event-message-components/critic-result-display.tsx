import React from "react";
import { useTranslation } from "react-i18next";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { useEventExpansionKey } from "#/components/features/chat/row-expansion-context";
import { usePersistentExpansion } from "#/stores/message-expansion-store";
import type {
  CriticResult,
  CriticFeature,
  CriticCategorizedFeatures,
} from "#/types/agent-server/core/base/critic";

/**
 * Normalize potentially malformed runtime scores before rendering.
 */
function normalizeScore(score: number): number {
  if (!Number.isFinite(score)) return 0;
  return Math.min(1, Math.max(0, score));
}

/**
 * Convert a normalized score (0-1) to a 5-star rating string.
 */
function getStarRating(normalizedScore: number): {
  filled: number;
  empty: number;
} {
  const filled = Math.round(normalizedScore * 5);
  return { filled, empty: 5 - filled };
}

/**
 * Get the color class for the star rating based on score.
 */
function getScoreColorClass(score: number): string {
  if (score >= 0.6) return "text-feedback-success";
  if (score >= 0.4) return "text-feedback-warning";
  return "text-feedback-error";
}

/**
 * Get the color class for an issue probability.
 */
function getIssueColorClass(probability: number): string {
  if (probability >= 0.7) return "text-feedback-error font-semibold";
  if (probability >= 0.5) return "text-feedback-warning";
  return "text-text-tertiary";
}

function isSettingsRecord(value: unknown): value is Record<string, unknown> {
  return (
    value !== null &&
    value !== undefined &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}

function getIterativeRefinementEnabled(
  agentSettings: Record<string, unknown> | null | undefined,
): boolean | null {
  const verification = agentSettings?.verification;
  if (!isSettingsRecord(verification)) {
    return null;
  }

  const value = verification.enable_iterative_refinement;
  return typeof value === "boolean" ? value : null;
}

/**
 * Renders a single issue feature with its probability.
 */
function FeatureItem({ feature }: { feature: CriticFeature }) {
  const percentage = Math.round(feature.probability * 100);
  const colorClass = getIssueColorClass(feature.probability);

  return (
    <span className="inline-flex items-center gap-1">
      <span className="text-contrast">{feature.display_name}</span>
      <span className={colorClass}>({percentage}%)</span>
    </span>
  );
}

/**
 * Renders a category of features (e.g., "Potential Issues", "Infrastructure").
 */
function FeatureCategory({
  label,
  features,
}: {
  label: string;
  features: CriticFeature[];
}) {
  if (!features || features.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-x-1 text-xs">
      <span className="font-semibold text-text-tertiary">{label}</span>
      {features.map((feature, i) => (
        <React.Fragment key={feature.name}>
          {i > 0 && <span className="text-text-tertiary">·</span>}
          <FeatureItem feature={feature} />
        </React.Fragment>
      ))}
    </div>
  );
}

/**
 * Renders the categorized features breakdown.
 */
function FeaturesBreakdown({
  categorized,
}: {
  categorized: CriticCategorizedFeatures;
}) {
  const { t } = useTranslation();

  const hasAgentIssues =
    categorized.agent_behavioral_issues &&
    categorized.agent_behavioral_issues.length > 0;
  const hasUserPatterns =
    categorized.user_followup_patterns &&
    categorized.user_followup_patterns.length > 0;
  const hasInfra =
    categorized.infrastructure_issues &&
    categorized.infrastructure_issues.length > 0;
  const hasOther = categorized.other && categorized.other.length > 0;

  if (!hasAgentIssues && !hasUserPatterns && !hasInfra && !hasOther) {
    return null;
  }

  return (
    <div className="flex flex-col gap-1 mt-1.5">
      {hasAgentIssues && (
        <FeatureCategory
          label={t(I18nKey.CRITIC$POTENTIAL_ISSUES)}
          features={categorized.agent_behavioral_issues!}
        />
      )}
      {hasInfra && (
        <FeatureCategory
          label={t(I18nKey.CRITIC$INFRASTRUCTURE)}
          features={categorized.infrastructure_issues!}
        />
      )}
      {hasUserPatterns && (
        <FeatureCategory
          label={t(I18nKey.CRITIC$LIKELY_FOLLOWUP)}
          features={categorized.user_followup_patterns!}
        />
      )}
      {hasOther && (
        <FeatureCategory
          label={t(I18nKey.CRITIC$OTHER)}
          features={categorized.other!}
        />
      )}
    </div>
  );
}

interface CriticResultDisplayProps {
  criticResult: CriticResult;
}

/**
 * Displays a critic evaluation result with star rating, score percentage,
 * and expandable categorized feature breakdown.
 */
export function CriticResultDisplay({
  criticResult,
}: CriticResultDisplayProps) {
  const { t } = useTranslation();
  const { data: settings } = useSettings();
  const expansionKey = useEventExpansionKey("critic-details");
  const [expanded, toggleExpanded] = usePersistentExpansion(
    expansionKey,
    false,
  );

  const normalizedScore = normalizeScore(criticResult.score);
  const { filled, empty } = getStarRating(normalizedScore);
  const colorClass = getScoreColorClass(normalizedScore);
  const percentage = (normalizedScore * 100).toFixed(1);
  const iterativeRefinementEnabled = getIterativeRefinementEnabled(
    settings?.agent_settings as Record<string, unknown> | null | undefined,
  );
  const showIterativeRefinementHint = iterativeRefinementEnabled === false;

  const categorized = criticResult.metadata?.categorized_features;
  const hasDetails =
    categorized != null &&
    ((categorized.agent_behavioral_issues ?? []).length > 0 ||
      (categorized.user_followup_patterns ?? []).length > 0 ||
      (categorized.infrastructure_issues ?? []).length > 0 ||
      (categorized.other ?? []).length > 0);

  return (
    <div className="border-l-2 border-border pl-2 my-2 py-1.5 text-sm">
      <div className="flex items-center gap-1">
        <span className="font-semibold text-text-tertiary text-xs">
          {t(I18nKey.CRITIC$SUCCESS_LIKELIHOOD_LABEL)}
        </span>
        <span
          className={`${colorClass} text-xs tracking-wide`}
          aria-label={`Score: ${percentage}%`}
        >
          {"★".repeat(filled)}
          {"☆".repeat(empty)}
        </span>
        <span className="text-text-tertiary text-xs">({percentage}%)</span>

        {hasDetails && (
          <button
            type="button"
            onClick={() => toggleExpanded()}
            className="cursor-pointer ml-1"
            aria-label={
              expanded
                ? t(I18nKey.BUTTON$COLLAPSE_DETAILS)
                : t(I18nKey.BUTTON$EXPAND_DETAILS)
            }
          >
            {expanded ? (
              <ArrowUp className="h-3 w-3 inline fill-text-tertiary" />
            ) : (
              <ArrowDown className="h-3 w-3 inline fill-text-tertiary" />
            )}
          </button>
        )}
      </div>

      {expanded && hasDetails && (
        <FeaturesBreakdown categorized={categorized!} />
      )}

      {showIterativeRefinementHint && (
        <p
          className="mt-1.5 text-xs leading-5 text-text-tertiary"
          data-testid="critic-iterative-refinement-hint"
        >
          {t(I18nKey.CRITIC$ITERATIVE_REFINEMENT_HINT)}
        </p>
      )}
    </div>
  );
}
