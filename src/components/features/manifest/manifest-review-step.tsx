import {
  interpolateText,
  type ManifestScope,
} from "#/manifests/manifest-template";
import type { ManifestReview } from "#/manifests/types";

export interface ManifestReviewStepProps {
  review: ManifestReview;
  scope: ManifestScope;
}

/**
 * Stage 7 — the plain-language summary the user confirms.
 *
 * The last cheap moment to catch a wrong answer, and the last point at which
 * nothing has been created yet. Every row's copy is manifest-authored; the host
 * only fills in the placeholders.
 */
export function ManifestReviewStep({ review, scope }: ManifestReviewStepProps) {
  return (
    <div className="flex flex-col gap-4" data-testid="manifest-review">
      {review.note && (
        <p className="text-sm text-[var(--oh-muted)]">{review.note}</p>
      )}
      <dl className="flex flex-col gap-3">
        {review.summary.map((row) => {
          const value = interpolateText(row.value, scope).trim();
          return (
            <div key={row.label} className="flex flex-col gap-0.5">
              <dt className="text-xs text-[var(--oh-muted)]">{row.label}</dt>
              <dd className="text-sm break-words">
                {value || review.emptyValueText || ""}
              </dd>
            </div>
          );
        })}
      </dl>
    </div>
  );
}
