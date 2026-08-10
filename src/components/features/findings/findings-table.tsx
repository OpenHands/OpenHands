/**
 * Findings table (desktop) + compact list (mobile).
 * @spec PROJETOSIN-188 — findings-table
 */

import { useTranslation } from "react-i18next";
import type { Finding } from "#/api/pentest/findings-types";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { I18nKey } from "#/i18n/declaration";
import {
  FindingSeverityBadge,
  FindingStatusBadge,
} from "./finding-severity-badge";
import {
  FindingsRowActions,
  type FindingsTriageAction,
} from "./findings-row-actions";

interface FindingsTableProps {
  findings: Finding[];
  canTriage: boolean;
  locale: string;
  onOpenDetail: (finding: Finding) => void;
  onTriageAction: (finding: Finding, action: FindingsTriageAction) => void;
}

export function FindingsTable({
  findings,
  canTriage,
  locale,
  onOpenDetail,
  onTriageAction,
}: FindingsTableProps) {
  const { t } = useTranslation("openhands");

  return (
    <div data-testid="findings-table" className="min-w-0">
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full min-w-[56rem] border-collapse text-left text-sm">
          <thead className="sticky top-0 bg-base">
            <tr className="border-b border-[var(--oh-border)] text-xs text-[var(--oh-text-tertiary)]">
              <th scope="col" className="px-2 py-2 font-medium">
                {t(I18nKey.FINDINGS$COL_SEVERITY)}
              </th>
              <th scope="col" className="px-2 py-2 font-medium">
                {t(I18nKey.FINDINGS$COL_TITLE)}
              </th>
              <th scope="col" className="px-2 py-2 font-medium">
                {t(I18nKey.FINDINGS$COL_ASSET)}
              </th>
              <th scope="col" className="px-2 py-2 font-medium">
                {t(I18nKey.FINDINGS$COL_ENDPOINT)}
              </th>
              <th scope="col" className="px-2 py-2 font-medium">
                {t(I18nKey.FINDINGS$COL_TOOL)}
              </th>
              <th scope="col" className="px-2 py-2 font-medium">
                {t(I18nKey.FINDINGS$COL_STATUS)}
              </th>
              <th scope="col" className="px-2 py-2 font-medium">
                {t(I18nKey.FINDINGS$COL_UPDATED)}
              </th>
              {canTriage ? (
                <th scope="col" className="px-2 py-2 font-medium">
                  {t(I18nKey.FINDINGS$COL_ACTIONS)}
                </th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {findings.map((finding) => (
              <tr
                key={finding.id}
                data-testid={`findings-row-${finding.id}`}
                tabIndex={0}
                className="cursor-pointer border-b border-[var(--oh-border)] hover:bg-[var(--oh-surface-raised)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--oh-color-primary)]"
                onClick={() => onOpenDetail(finding)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onOpenDetail(finding);
                  }
                }}
              >
                <td className="px-2 py-3 align-middle">
                  <FindingSeverityBadge severity={finding.severity} />
                </td>
                <td className="max-w-[16rem] truncate px-2 py-3 align-middle text-white">
                  {finding.title}
                </td>
                <td
                  className="max-w-[10rem] truncate px-2 py-3 align-middle text-[var(--oh-text-secondary)]"
                  title={finding.asset ?? undefined}
                >
                  {finding.asset ?? "—"}
                </td>
                <td
                  className="max-w-[12rem] truncate px-2 py-3 align-middle font-mono text-xs text-[var(--oh-text-secondary)]"
                  title={finding.endpoint ?? undefined}
                >
                  {finding.endpoint ?? "—"}
                </td>
                <td className="px-2 py-3 align-middle">
                  <span className={extensionModuleCardPillClassName}>
                    {finding.source_tool}
                  </span>
                </td>
                <td className="px-2 py-3 align-middle">
                  <FindingStatusBadge status={finding.status} />
                </td>
                <td className="whitespace-nowrap px-2 py-3 align-middle text-[var(--oh-text-tertiary)]">
                  {formatRelativeTime(finding.updated_at, locale, t)}
                </td>
                {canTriage ? (
                  <td className="px-2 py-3 align-middle">
                    <FindingsRowActions
                      findingTitle={finding.title}
                      onAction={(action) => onTriageAction(finding, action)}
                    />
                  </td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <ul className="flex flex-col gap-2 md:hidden">
        {findings.map((finding) => (
          <li key={finding.id}>
            <div
              data-testid={`findings-row-${finding.id}`}
              className="flex w-full items-start gap-2 rounded-xl border border-[var(--oh-border)] p-3 hover:bg-[var(--oh-surface-raised)]"
            >
              <button
                type="button"
                className="flex min-w-0 flex-1 flex-col gap-1 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--oh-color-primary)]"
                onClick={() => onOpenDetail(finding)}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <FindingSeverityBadge severity={finding.severity} />
                  <FindingStatusBadge status={finding.status} />
                </div>
                <span className="truncate text-sm text-white">
                  {finding.title}
                </span>
                <span className="truncate font-mono text-xs text-[var(--oh-text-tertiary)]">
                  {finding.endpoint ?? finding.asset ?? finding.source_tool}
                </span>
              </button>
              {canTriage ? (
                <FindingsRowActions
                  findingTitle={finding.title}
                  onAction={(action) => onTriageAction(finding, action)}
                />
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
