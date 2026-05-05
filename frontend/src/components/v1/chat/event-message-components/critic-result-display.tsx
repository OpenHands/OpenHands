import React from "react";
import { useTranslation } from "react-i18next";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import type {
  CriticResult,
  CriticFeature,
  CriticCategorizedFeatures,
} from "#/types/v1/core/base/critic";

/**
 * Convert a score (0-1) to a 5-star rating string.
 */
function getStarRating(score: number): { filled: number; empty: number } {
  const filled = Math.round(score * 5);
  return { filled, empty: 5 - filled };
}

/**
 * Get the color class for the star rating based on score.
 */
function getScoreColorClass(score: number): string {
  if (score >= 0.6) return "text-green-400";
  if (score >= 0.4) return "text-yellow-400";
  return "text-red-400";
}

/**
 * Get the color class for an issue probability.
 */
function getIssueColorClass(probability: number): string {
  if (probability >= 0.7) return "text-red-400 font-semibold";
  if (probability >= 0.5) return "text-yellow-400";
  return "text-neutral-400";
}

function getIterativeRefinementEnabled(
  agentSettings: Record<string, unknown> | null | undefined,
): boolean | null {
  const verification = agentSettings?.verification;
  if (
    verification == null ||
    typeof verification !== "object" ||
    Array.isArray(verification)
  ) {
    return null;
  }

  const value = (verification as Record<string, unknown>)
    .enable_iterative_refinement;
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
      <span className="text-neutral-200">{feature.display_name}</span>
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
      <span className="font-semibold text-neutral-300">{label}</span>
      {features.map((feature, i) => (
        <React.Fragment key={feature.name}>
          {i > 0 && <span className="text-neutral-500">·</span>}
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
  const [expanded, setExpanded] = React.useState(false);

  const { filled, empty } = getStarRating(criticResult.score);
  const colorClass = getScoreColorClass(criticResult.score);
  const percentage = (criticResult.score * 100).toFixed(1);
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
    <div className="border-l-2 border-neutral-600 pl-2 my-2 py-1.5 text-sm">
      <div className="flex items-center gap-1">
        <span className="font-semibold text-neutral-300 text-xs">
          {t(I18nKey.CRITIC$SUCCESS_LIKELIHOOD_LABEL)}
        </span>
        <span className={`${colorClass} text-xs tracking-wide`}>
          {"★".repeat(filled)}
          {"☆".repeat(empty)}
        </span>
        <span className="text-neutral-500 text-xs">({percentage}%)</span>

        {hasDetails && (
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="cursor-pointer ml-1"
            aria-label={expanded ? "Collapse details" : "Expand details"}
          >
            {expanded ? (
              <ArrowUp className="h-3 w-3 inline fill-neutral-400" />
            ) : (
              <ArrowDown className="h-3 w-3 inline fill-neutral-400" />
            )}
          </button>
        )}
      </div>

      {expanded && hasDetails && (
        <FeaturesBreakdown categorized={categorized!} />
      )}

      {showIterativeRefinementHint && (
        <p
          className="mt-1.5 text-xs leading-5 text-neutral-500"
          data-testid="critic-iterative-refinement-hint"
        >
          {t(I18nKey.CRITIC$ITERATIVE_REFINEMENT_HINT)}
        </p>
      )}
    </div>
  );
}
