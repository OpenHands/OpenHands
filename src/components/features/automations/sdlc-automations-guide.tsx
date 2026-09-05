import { useMemo, useState, type ComponentType } from "react";
import {
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  Code2,
  FlaskConical,
  LockKeyhole,
  Rocket,
  SearchCheck,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  AUTOMATION_CATALOG,
  type RecommendedAutomation,
} from "@openhands/extensions/automations";
import { INTEGRATION_CATALOG } from "@openhands/extensions/integrations";
import { I18nKey } from "#/i18n/declaration";
import type { MCPServerConfig } from "#/types/mcp-server";
import type { Automation } from "#/types/automation";
import { getRequiredIntegrationIds } from "#/utils/automation-catalog";
import { normalizeAutomationKey } from "#/utils/recommended-automation-rail";
import {
  findInstalledEntryMatch,
  getMarketplaceEntryById,
} from "#/utils/mcp-marketplace-utils";
import { cn } from "#/utils/utils";

export type SdlcPhaseId =
  | "plan"
  | "implement"
  | "verify"
  | "review"
  | "release";

type SdlcPhaseStatus = "complete" | "partial" | "available" | "blocked";

interface SdlcPhaseDefinition {
  id: SdlcPhaseId;
  Icon: ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  automationIds: readonly string[];
}

/**
 * This is presentation metadata for the Canvas roadmap, not an automation
 * definition. IDs that are absent from the installed extensions version are
 * ignored so catalog releases can add the matching workflow independently.
 */
const SDLC_PHASES: readonly SdlcPhaseDefinition[] = [
  {
    id: "plan",
    Icon: ClipboardList,
    automationIds: ["linear-triage-assistant", "slack-standup-digest"],
  },
  {
    id: "implement",
    Icon: Code2,
    automationIds: [
      "github-issue-to-pr",
      "linear-issue-to-github-pr",
      "linear-issue-to-gitlab-mr",
      "linear-issue-to-bitbucket-pr",
      "jira-issue-to-pr",
      "jira-issue-to-gitlab-mr",
      "jira-issue-to-bitbucket-pr",
    ],
  },
  {
    id: "verify",
    Icon: FlaskConical,
    automationIds: ["qa-changes", "incident-retrospective-drafter"],
  },
  {
    id: "review",
    Icon: SearchCheck,
    automationIds: ["github-pr-reviewer", "github-repo-monitor"],
  },
  {
    id: "release",
    Icon: Rocket,
    automationIds: ["upstream-fork-sync", "github-agents-md-maintainer"],
  },
];

const PHASE_COPY = {
  plan: {
    label: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_PLAN,
    description: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_PLAN_DESCRIPTION,
  },
  implement: {
    label: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_IMPLEMENT,
    description: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_IMPLEMENT_DESCRIPTION,
  },
  verify: {
    label: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_VERIFY,
    description: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_VERIFY_DESCRIPTION,
  },
  review: {
    label: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_REVIEW,
    description: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_REVIEW_DESCRIPTION,
  },
  release: {
    label: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_RELEASE,
    description: I18nKey.SDLC_AUTOMATIONS_GUIDE$PHASE_RELEASE_DESCRIPTION,
  },
} as const;

const STATUS_COPY = {
  complete: I18nKey.SDLC_AUTOMATIONS_GUIDE$STATUS_COMPLETE,
  partial: I18nKey.SDLC_AUTOMATIONS_GUIDE$STATUS_PARTIAL,
  available: I18nKey.SDLC_AUTOMATIONS_GUIDE$STATUS_AVAILABLE,
  blocked: I18nKey.RECOMMENDED_AUTOMATIONS$NEEDS_SETUP,
} as const;

export interface SdlcAutomationOpportunity {
  automation: RecommendedAutomation;
  installedAutomation: Automation | null;
  missingIntegrations: string[];
}

export interface SdlcPhaseState extends SdlcPhaseDefinition {
  status: SdlcPhaseStatus;
  opportunities: SdlcAutomationOpportunity[];
}

function installedAutomationFor(
  entry: RecommendedAutomation,
  installedAutomations: readonly Automation[],
): Automation | null {
  const matchKeys = [entry.id, entry.name, entry.skill]
    .filter((value): value is string => Boolean(value))
    .map(normalizeAutomationKey);

  return (
    installedAutomations.find((automation) =>
      matchKeys.includes(normalizeAutomationKey(automation.name)),
    ) ?? null
  );
}

function missingIntegrationNames(
  automation: RecommendedAutomation,
  installedServers: MCPServerConfig[],
): string[] {
  return getRequiredIntegrationIds(automation).flatMap((id) => {
    const entry = getMarketplaceEntryById(id, INTEGRATION_CATALOG);
    if (entry && findInstalledEntryMatch(entry, installedServers)) return [];
    return [entry?.name ?? id];
  });
}

export function buildSdlcPhaseStates(
  installedAutomations: readonly Automation[],
  installedServers: MCPServerConfig[],
): SdlcPhaseState[] {
  const catalogById = new Map(
    AUTOMATION_CATALOG.map((automation) => [automation.id, automation]),
  );

  return SDLC_PHASES.map((phase) => {
    const opportunities = phase.automationIds.flatMap((id) => {
      const automation = catalogById.get(id);
      if (!automation) return [];
      return [
        {
          automation,
          installedAutomation: installedAutomationFor(
            automation,
            installedAutomations,
          ),
          missingIntegrations: missingIntegrationNames(
            automation,
            installedServers,
          ),
        },
      ];
    });
    const hasEnabled = opportunities.some(
      ({ installedAutomation }) => installedAutomation?.enabled,
    );
    const hasInstalled = opportunities.some(
      ({ installedAutomation }) => installedAutomation !== null,
    );
    const hasReady = opportunities.some(
      ({ installedAutomation, missingIntegrations }) =>
        !installedAutomation && missingIntegrations.length === 0,
    );

    return {
      ...phase,
      opportunities,
      status: hasEnabled
        ? "complete"
        : hasInstalled
          ? "partial"
          : hasReady
            ? "available"
            : "blocked",
    };
  });
}

function statusIcon(status: SdlcPhaseStatus) {
  if (status === "complete") {
    return <CheckCircle2 className="size-3.5" aria-hidden />;
  }
  if (status === "blocked") {
    return <LockKeyhole className="size-3.5" aria-hidden />;
  }
  return <span className="size-2 rounded-full bg-current" aria-hidden />;
}

function opportunityStatus(
  opportunity: SdlcAutomationOpportunity,
  translate: ReturnType<typeof useTranslation>["t"],
) {
  if (opportunity.installedAutomation?.enabled) {
    return translate(I18nKey.AUTOMATIONS$ACTIVE);
  }
  if (opportunity.installedAutomation) {
    return translate(I18nKey.AUTOMATIONS$INACTIVE);
  }
  if (opportunity.missingIntegrations.length > 0) {
    return translate(I18nKey.SDLC_AUTOMATIONS_GUIDE$MISSING_INTEGRATIONS, {
      integrations: opportunity.missingIntegrations.join(", "),
    });
  }
  return translate(I18nKey.RECOMMENDED_AUTOMATIONS$MINUTES, {
    count: opportunity.automation.estimatedSetupMinutes,
  });
}

interface SdlcAutomationsGuideProps {
  installedAutomations: readonly Automation[];
  installedServers: MCPServerConfig[];
  onSelect: (automation: RecommendedAutomation) => void;
  onOpenInstalled: (automation: Automation) => void;
}

export function SdlcAutomationsGuide({
  installedAutomations,
  installedServers,
  onSelect,
  onOpenInstalled,
}: SdlcAutomationsGuideProps) {
  const { t } = useTranslation("openhands");
  const [selectedPhaseId, setSelectedPhaseId] = useState<SdlcPhaseId>("plan");
  const phases = useMemo(
    () => buildSdlcPhaseStates(installedAutomations, installedServers),
    [installedAutomations, installedServers],
  );
  const selectedPhase =
    phases.find((phase) => phase.id === selectedPhaseId) ?? phases[0];
  const completedUnits = phases.reduce(
    (total, phase) =>
      total +
      (phase.status === "complete" ? 1 : phase.status === "partial" ? 0.5 : 0),
    0,
  );
  const progress = Math.round((completedUnits / phases.length) * 100);
  const startedCount = phases.filter(
    (phase) => phase.status === "complete" || phase.status === "partial",
  ).length;
  const suggestedPhase =
    phases.find((phase) => phase.status === "available") ??
    phases.find((phase) => phase.status !== "complete") ??
    phases[phases.length - 1];

  if (!selectedPhase) return null;

  return (
    <section
      data-testid="sdlc-automations-guide"
      className="overflow-hidden rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface)]"
      aria-labelledby="sdlc-automations-guide-title"
    >
      <div className="border-b border-[var(--oh-border-subtle)] p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-2xl">
            <h2
              id="sdlc-automations-guide-title"
              className="text-base font-semibold text-content"
            >
              {t(I18nKey.SDLC_AUTOMATIONS_GUIDE$TITLE)}
            </h2>
            <p className="mt-1 text-sm leading-relaxed text-muted">
              {t(I18nKey.SDLC_AUTOMATIONS_GUIDE$DESCRIPTION)}
            </p>
          </div>
          <span className="rounded-full bg-[var(--oh-surface-raised)] px-3 py-1 text-xs font-medium text-content">
            {t(I18nKey.SDLC_AUTOMATIONS_GUIDE$PROGRESS, { progress })}
          </span>
        </div>

        <div
          className="mt-4 h-1.5 overflow-hidden rounded-full bg-[var(--oh-surface-raised)]"
          role="progressbar"
          aria-label={t(I18nKey.SDLC_AUTOMATIONS_GUIDE$TITLE)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress}
        >
          <div
            className="h-full rounded-full bg-[var(--oh-interactive)] transition-[width] motion-reduce:transition-none"
            style={{ width: `${String(progress)}%` }}
          />
        </div>
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1.1fr)_minmax(300px,0.9fr)]">
        <div
          className="grid grid-cols-2 gap-2 border-b border-[var(--oh-border-subtle)] p-3 sm:grid-cols-5 lg:grid-cols-1 lg:border-r lg:border-b-0"
          aria-label={t(I18nKey.SDLC_AUTOMATIONS_GUIDE$TITLE)}
        >
          {phases.map((phase, index) => {
            const copy = PHASE_COPY[phase.id];
            const isSelected = phase.id === selectedPhase.id;
            return (
              <button
                key={phase.id}
                type="button"
                data-testid={`sdlc-phase-${phase.id}`}
                aria-pressed={isSelected}
                onClick={() => setSelectedPhaseId(phase.id)}
                className={cn(
                  "group flex min-w-0 items-center gap-3 rounded-lg border px-3 py-3 text-left transition-colors",
                  isSelected
                    ? "border-[var(--oh-interactive)] bg-[var(--oh-interactive-selected)]"
                    : "border-transparent hover:bg-[var(--oh-interactive-hover)]",
                )}
              >
                <span
                  className={cn(
                    "flex size-8 shrink-0 items-center justify-center rounded-lg bg-[var(--oh-surface-raised)] text-muted",
                    isSelected && "text-content",
                  )}
                >
                  <phase.Icon className="size-4" aria-hidden />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-1.5 text-sm font-medium text-content">
                    <span className="text-xs text-muted">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="truncate">{t(copy.label)}</span>
                  </span>
                  <span
                    className={cn(
                      "mt-1 flex items-center gap-1.5 text-xs",
                      phase.status === "complete" && "text-success",
                      phase.status === "partial" && "text-warning",
                      phase.status === "available" && "text-muted",
                      phase.status === "blocked" && "text-tertiary-alt",
                    )}
                  >
                    {statusIcon(phase.status)}
                    {t(STATUS_COPY[phase.status])}
                  </span>
                </span>
              </button>
            );
          })}
        </div>

        <div className="min-w-0 p-5">
          <div className="flex items-center gap-3">
            <selectedPhase.Icon className="size-5 text-muted" aria-hidden />
            <div className="min-w-0">
              <h3 className="text-sm font-semibold text-content">
                {t(PHASE_COPY[selectedPhase.id].label)}
              </h3>
              <p className="mt-0.5 text-xs leading-relaxed text-muted">
                {t(PHASE_COPY[selectedPhase.id].description)}
              </p>
            </div>
          </div>

          <div className="mt-4 rounded-lg bg-[var(--oh-surface-raised)] px-3 py-2 text-xs text-muted">
            {t(
              startedCount > 0
                ? I18nKey.SDLC_AUTOMATIONS_GUIDE$NEXT_STEP
                : I18nKey.SDLC_AUTOMATIONS_GUIDE$START_STEP,
              { phase: t(PHASE_COPY[suggestedPhase.id].label) },
            )}
          </div>

          <div className="mt-4 flex flex-col gap-2">
            {selectedPhase.opportunities.slice(0, 3).map((opportunity) => {
              const installed = opportunity.installedAutomation;
              return (
                <div
                  key={opportunity.automation.id}
                  data-testid={`sdlc-opportunity-${opportunity.automation.id}`}
                  className="flex items-center gap-3 rounded-lg border border-[var(--oh-border-subtle)] p-3"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-content">
                      {opportunity.automation.name}
                    </p>
                    <p className="mt-0.5 truncate text-xs text-muted">
                      {opportunityStatus(opportunity, t)}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      installed
                        ? onOpenInstalled(installed)
                        : onSelect(opportunity.automation)
                    }
                    className="flex shrink-0 items-center gap-1 rounded-md px-2.5 py-1.5 text-xs font-medium text-content hover:bg-[var(--oh-interactive-hover)]"
                  >
                    {t(
                      installed
                        ? I18nKey.AUTOMATIONS$EDIT
                        : I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON,
                    )}
                    <ArrowRight className="size-3.5" aria-hidden />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
