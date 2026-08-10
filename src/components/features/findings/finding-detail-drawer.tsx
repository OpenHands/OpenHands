/**
 * Finding detail drawer (right overlay).
 * @spec PROJETOSIN-188 — finding-detail-drawer
 */

import React from "react";
import { useTranslation } from "react-i18next";
import type { Finding } from "#/api/pentest/findings-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useNavigation } from "#/context/navigation-context";
import { useFindingDetail } from "#/hooks/query/use-findings";
import { I18nKey } from "#/i18n/declaration";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import {
  FindingSeverityBadge,
  FindingStatusBadge,
} from "./finding-severity-badge";
import {
  FindingsRowActions,
  type FindingsTriageAction,
} from "./findings-row-actions";

interface FindingDetailDrawerProps {
  findingId: string | null;
  canTriage: boolean;
  locale: string;
  onClose: () => void;
  onTriageAction: (finding: Finding, action: FindingsTriageAction) => void;
}

export function FindingDetailDrawer({
  findingId,
  canTriage,
  locale,
  onClose,
  onTriageAction,
}: FindingDetailDrawerProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const [evidenceOpen, setEvidenceOpen] = React.useState(false);
  const open = Boolean(findingId);
  const detailQuery = useFindingDetail(findingId, { enabled: open });

  React.useEffect(() => {
    setEvidenceOpen(false);
  }, [findingId]);

  React.useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const finding = detailQuery.data;
  const evidence = finding?.evidence ?? null;
  const conversationId =
    evidence && typeof evidence.conversation_id === "string"
      ? evidence.conversation_id
      : null;
  const eventId =
    evidence && typeof evidence.event_id === "string"
      ? evidence.event_id
      : null;
  const canOpenStream = Boolean(conversationId && eventId);

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        type="button"
        className="absolute inset-0 bg-black/50"
        aria-label={t(I18nKey.FINDINGS$DRAWER_CLOSE)}
        onClick={onClose}
      />
      <aside
        data-testid="finding-detail-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={t(I18nKey.FINDINGS$DRAWER_TITLE)}
        className="relative z-10 flex h-full w-full max-w-md flex-col border-l border-[var(--oh-border)] bg-[var(--oh-surface)] shadow-xl md:max-w-[28rem]"
      >
        <div className="flex items-center justify-between border-b border-[var(--oh-border)] px-4 py-3">
          <h2 className="text-sm font-medium text-white">
            {t(I18nKey.FINDINGS$DRAWER_TITLE)}
          </h2>
          <button
            type="button"
            className="rounded-md px-2 py-1 text-sm text-[var(--oh-text-secondary)] hover:bg-[var(--oh-surface-raised)]"
            aria-label={t(I18nKey.FINDINGS$DRAWER_CLOSE)}
            onClick={onClose}
          >
            {}
            <span aria-hidden="true">×</span>
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          {detailQuery.isLoading ? (
            <div
              data-testid="finding-detail-loading"
              aria-busy="true"
              className="flex flex-col gap-3"
            >
              {Array.from({ length: 5 }).map((_, i) => (
                <div
                  key={i}
                  className="h-8 animate-pulse rounded-md bg-[var(--oh-surface-raised)]"
                />
              ))}
            </div>
          ) : null}

          {detailQuery.isError ? (
            <div role="alert" className="text-sm text-[var(--oh-color-danger)]">
              {t(I18nKey.FINDINGS$ERROR)}
              <div className="mt-3">
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={() => void detailQuery.refetch()}
                >
                  {t(I18nKey.FINDINGS$ERROR_RETRY)}
                </BrandButton>
              </div>
            </div>
          ) : null}

          {finding ? (
            <div className="flex flex-col gap-4">
              <div className="flex flex-wrap gap-2">
                <FindingSeverityBadge severity={finding.severity} />
                <FindingStatusBadge status={finding.status} />
              </div>
              <h3 className="text-lg font-medium text-white">
                {finding.title}
              </h3>
              <dl className="grid grid-cols-1 gap-2 text-sm">
                <MetaRow
                  label={t(I18nKey.FINDINGS$COL_ASSET)}
                  value={finding.asset}
                />
                <MetaRow
                  label={t(I18nKey.FINDINGS$COL_ENDPOINT)}
                  value={finding.endpoint}
                  mono
                />
                <div>
                  <dt className="text-xs text-[var(--oh-text-tertiary)]">
                    {t(I18nKey.FINDINGS$COL_TOOL)}
                  </dt>
                  <dd>
                    <span className={extensionModuleCardPillClassName}>
                      {finding.source_tool}
                    </span>
                  </dd>
                </div>
                <MetaRow
                  label={t(I18nKey.FINDINGS$COL_UPDATED)}
                  value={formatRelativeTime(finding.updated_at, locale, t)}
                />
              </dl>

              {finding.description ? (
                <p className="whitespace-pre-wrap text-sm text-[var(--oh-text-secondary)]">
                  {finding.description}
                </p>
              ) : null}

              <div>
                <button
                  type="button"
                  data-testid="finding-evidence-toggle"
                  aria-expanded={evidenceOpen}
                  className="text-sm text-[var(--oh-color-primary)] underline-offset-2 hover:underline"
                  onClick={() => setEvidenceOpen((value) => !value)}
                >
                  {evidenceOpen
                    ? t(I18nKey.FINDINGS$EVIDENCE_COLLAPSE)
                    : t(I18nKey.FINDINGS$EVIDENCE_EXPAND)}
                </button>
                {evidenceOpen ? (
                  <div className="mt-2 rounded-md border border-[var(--oh-border)] bg-base-secondary p-3">
                    <p className="mb-2 text-xs font-medium text-[var(--oh-text-tertiary)]">
                      {t(I18nKey.FINDINGS$EVIDENCE_TITLE)}
                    </p>
                    {canOpenStream ? (
                      <BrandButton
                        type="button"
                        variant="secondary"
                        testId="finding-evidence-open-stream"
                        onClick={() =>
                          navigate(
                            `/conversations/${conversationId}?event=${eventId}`,
                          )
                        }
                      >
                        {t(I18nKey.FINDINGS$EVIDENCE_OPEN_STREAM)}
                      </BrandButton>
                    ) : (
                      <p className="mb-2 text-xs text-[var(--oh-text-tertiary)]">
                        {t(I18nKey.FINDINGS$EVIDENCE_NO_LINK)}
                      </p>
                    )}
                    <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all font-mono text-xs text-[var(--oh-text-secondary)]">
                      {JSON.stringify(evidence ?? {}, null, 2)}
                    </pre>
                  </div>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>

        {canTriage && finding ? (
          <div className="flex items-center justify-end gap-2 border-t border-[var(--oh-border)] px-4 py-3">
            <FindingsRowActions
              findingTitle={finding.title}
              onAction={(action) => onTriageAction(finding, action)}
            />
          </div>
        ) : null}
      </aside>
    </div>
  );
}

function MetaRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string | null | undefined;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="text-xs text-[var(--oh-text-tertiary)]">{label}</dt>
      <dd
        className={
          mono
            ? "font-mono text-xs text-[var(--oh-text-secondary)]"
            : "text-[var(--oh-text-secondary)]"
        }
      >
        {value || "—"}
      </dd>
    </div>
  );
}
