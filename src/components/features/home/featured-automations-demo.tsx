import { Plus, Sparkles } from "lucide-react";
import { useState } from "react";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { cn } from "#/utils/utils";

type AutomationHealth = "success" | "failed" | "running";

interface DemoAutomation {
  id: string;
  name: string;
  health: AutomationHealth;
  lastRun: string;
  detail: string;
  result: string;
  nextRun: string;
  agentMessage: string;
  error?: string;
  conversationId: string;
}

const DEMO_AUTOMATIONS: DemoAutomation[] = [
  {
    id: "pr-review",
    name: "PR reviewer",
    health: "success",
    lastRun: "Succeeded 12 min ago",
    detail: "Reviewed #16182 and left 3 actionable comments.",
    result: "3 suggestions posted · 1 security check passed",
    nextRun: "On next pull request",
    agentMessage:
      "I reviewed the pull request and posted three suggestions. The security check completed without findings.",
    conversationId: "automation-pr-review-16182",
  },
  {
    id: "issue-triage",
    name: "Issue triage",
    health: "failed",
    lastRun: "Failed 28 min ago",
    detail: "The run completed, but the model provider rejected one request.",
    result: "18 issues classified · 1 retry needed",
    nextRun: "In 4 minutes",
    agentMessage:
      "I classified 18 incoming issues and queued the remaining repository lookup for retry.",
    error:
      "Model provider rejected the repository lookup request: rate limit exceeded.",
    conversationId: "automation-issue-triage-20260801",
  },
  {
    id: "weekly-digest",
    name: "Weekly workflow digest",
    health: "running",
    lastRun: "Running now",
    detail: "Collecting run quality, cost, and failure patterns for the team.",
    result: "Preparing this week's automation health summary",
    nextRun: "Monday at 9:00 AM",
    agentMessage:
      "I am collecting run quality, cost, and failure patterns for this week's summary.",
    conversationId: "automation-weekly-digest-20260801",
  },
  {
    id: "repo-monitor",
    name: "Repository monitor",
    health: "success",
    lastRun: "Succeeded 1 hr ago",
    detail: "Found no new dependency or workflow failures.",
    result: "42 checks scanned · no action required",
    nextRun: "In 59 minutes",
    agentMessage:
      "I scanned the configured repositories and found no dependency or workflow failures requiring action.",
    conversationId: "automation-repository-monitor-20260801",
  },
];

const HEALTH_LABEL: Record<AutomationHealth, string> = {
  success: "Last run succeeded",
  failed: "Last run failed",
  running: "Run in progress",
};

const DEMO_COPY = {
  title: "Automations",
  openAutomations: "Open automations",
  featured: "Featured",
  manageAutomations: "Add or manage automations",
  recentConversation: "Open most recent conversation",
};

function AutomationStatus({ health }: { health: AutomationHealth }) {
  if (health === "success") {
    return (
      <svg
        viewBox="0 0 12 12"
        className="h-2.5 w-2.5 stroke-[var(--oh-status-success)]"
        fill="none"
        strokeWidth={2.25}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-label={HEALTH_LABEL[health]}
      >
        <path d="M2.5 6.5 5 9l4.5-5.5" />
      </svg>
    );
  }

  return (
    <span
      aria-label={HEALTH_LABEL[health]}
      className={cn(
        "h-1.5 w-1.5 rounded-full",
        health === "failed"
          ? "bg-[var(--oh-status-error)]"
          : "animate-pulse bg-[var(--oh-status-success)]",
      )}
    />
  );
}

function AutomationTooltip({ automation }: { automation: DemoAutomation }) {
  return (
    <div className="w-64 space-y-2 p-1 text-left text-white">
      <p className="font-semibold">{automation.name}</p>
      <p className="text-[var(--oh-text-secondary)]">{automation.detail}</p>
      <div className="flex items-center justify-between border-t border-[var(--oh-border-subtle)] pt-2 text-[11px] text-[var(--oh-text-secondary)]">
        <span>{automation.lastRun}</span>
        <span className="flex items-center gap-1">
          <AutomationStatus health={automation.health} />
          {HEALTH_LABEL[automation.health]}
        </span>
      </div>
    </div>
  );
}

/**
 * Product-review prototype, deliberately gated behind VITE_FEATURED_AUTOMATIONS_DEMO.
 * It makes the home-page interaction tangible without presenting mock runs as live data.
 */
export function FeaturedAutomationsDemo() {
  const [featured, setFeatured] = useState<DemoAutomation[]>([]);

  const addFeatured = (automation: DemoAutomation) => {
    setFeatured((current) =>
      current.some((item) => item.id === automation.id)
        ? current
        : [...current, automation],
    );
  };

  return (
    <section
      aria-labelledby="featured-automations-heading"
      data-testid="featured-automations-demo"
      className="mx-auto w-full max-w-5xl rounded-xl border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)] p-4"
    >
      <h2
        id="featured-automations-heading"
        className="mb-3 text-sm font-medium text-[var(--oh-foreground)]"
      >
        {DEMO_COPY.title}
      </h2>

      <div
        className="flex flex-wrap gap-2"
        aria-label={DEMO_COPY.openAutomations}
      >
        {DEMO_AUTOMATIONS.map((automation) => {
          const isFeatured = featured.some((item) => item.id === automation.id);

          return (
            <StyledTooltip
              key={automation.id}
              content={<AutomationTooltip automation={automation} />}
              placement="bottom"
              tooltipClassName="!max-w-none !border !border-[var(--oh-border-subtle)] !bg-[var(--oh-surface)]"
            >
              <button
                type="button"
                aria-label={automation.name}
                aria-pressed={isFeatured}
                onClick={() => addFeatured(automation)}
                className={cn(
                  "inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm text-[var(--oh-foreground)] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]",
                  isFeatured
                    ? "border-[var(--oh-focus)] bg-[var(--oh-interactive-selected)] shadow-inner"
                    : "border-[var(--oh-border)] bg-[var(--oh-surface-raised)] hover:bg-[var(--oh-interactive-hover)]",
                )}
              >
                <AutomationStatus health={automation.health} />
                {automation.name}
              </button>
            </StyledTooltip>
          );
        })}
        <StyledTooltip
          content={DEMO_COPY.manageAutomations}
          placement="bottom"
          tooltipClassName="!border !border-[var(--oh-border-subtle)] !bg-[var(--oh-surface)]"
        >
          <a
            href="/automations"
            aria-label={DEMO_COPY.manageAutomations}
            className="inline-flex items-center justify-center rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] px-3 py-2 text-[var(--oh-foreground)] transition-colors hover:bg-[var(--oh-interactive-hover)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
          >
            <Plus size={16} aria-hidden="true" />
          </a>
        </StyledTooltip>
      </div>

      {featured.length > 0 ? (
        <div className="mt-4 border-t border-[var(--oh-border-subtle)] pt-4">
          <div className="mb-3 flex items-center gap-2">
            <Sparkles
              size={16}
              className="text-[var(--oh-status-success)]"
              aria-hidden="true"
            />
            <h3 className="text-sm font-medium text-[var(--oh-foreground)]">
              {DEMO_COPY.featured}
            </h3>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {featured.map((automation) => {
              return (
                <article
                  key={automation.id}
                  className="rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] p-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h4 className="font-medium text-[var(--oh-foreground)]">
                        {automation.name}
                      </h4>
                      <p className="mt-1 text-sm text-[var(--oh-text-secondary)]">
                        {automation.result}
                      </p>
                    </div>
                    <AutomationStatus health={automation.health} />
                  </div>
                  <div className="mt-4 flex items-center gap-1 text-xs text-[var(--oh-text-secondary)]">
                    <span>
                      {automation.lastRun} · {automation.nextRun}
                    </span>
                    <a
                      href={`/conversations/${automation.conversationId}`}
                      aria-label={DEMO_COPY.recentConversation}
                      className="text-[var(--oh-foreground)] underline underline-offset-4 hover:text-[var(--oh-text-secondary)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
                    >
                      &gt;&gt;
                    </a>
                  </div>
                  <div className="mt-3 max-h-24 space-y-2 overflow-y-auto rounded-md border border-[var(--oh-border-subtle)] bg-[var(--oh-surface)] p-3 text-sm">
                    <p className="text-[var(--oh-text-secondary)]">
                      {automation.agentMessage}
                    </p>
                    {automation.error ? (
                      <p className="text-[var(--oh-status-error)]">
                        {automation.error}
                      </p>
                    ) : null}
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
}
