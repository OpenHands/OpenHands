import { CheckCircle2, CircleAlert, Clock3, Sparkles } from "lucide-react";
import { useState } from "react";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { cn } from "#/utils/utils";

type AutomationHealth = "healthy" | "attention" | "running";

interface DemoAutomation {
  id: string;
  name: string;
  health: AutomationHealth;
  lastRun: string;
  detail: string;
  result: string;
  nextRun: string;
}

const DEMO_AUTOMATIONS: DemoAutomation[] = [
  {
    id: "pr-review",
    name: "PR reviewer",
    health: "healthy",
    lastRun: "Succeeded 12 min ago",
    detail: "Reviewed #16182 and left 3 actionable comments.",
    result: "3 suggestions posted · 1 security check passed",
    nextRun: "On next pull request",
  },
  {
    id: "issue-triage",
    name: "Issue triage",
    health: "attention",
    lastRun: "Needs attention 28 min ago",
    detail: "The run completed, but the model provider rejected one request.",
    result: "18 issues classified · 1 retry needed",
    nextRun: "In 4 minutes",
  },
  {
    id: "weekly-digest",
    name: "Weekly workflow digest",
    health: "running",
    lastRun: "Running now",
    detail: "Collecting run quality, cost, and failure patterns for the team.",
    result: "Preparing this week's automation health summary",
    nextRun: "Monday at 9:00 AM",
  },
  {
    id: "repo-monitor",
    name: "Repository monitor",
    health: "healthy",
    lastRun: "Succeeded 1 hr ago",
    detail: "Found no new dependency or workflow failures.",
    result: "42 checks scanned · no action required",
    nextRun: "In 59 minutes",
  },
];

const HEALTH_STYLE: Record<
  AutomationHealth,
  { dot: string; icon: typeof CheckCircle2; label: string }
> = {
  healthy: {
    dot: "bg-emerald-500",
    icon: CheckCircle2,
    label: "Last run succeeded",
  },
  attention: {
    dot: "bg-amber-400",
    icon: CircleAlert,
    label: "Last run needs attention",
  },
  running: { dot: "bg-sky-500", icon: Clock3, label: "Run in progress" },
};

function AutomationTooltip({ automation }: { automation: DemoAutomation }) {
  const health = HEALTH_STYLE[automation.health];

  return (
    <div className="w-64 space-y-2 p-1 text-left">
      <p className="font-semibold">{automation.name}</p>
      <p className="text-slate-600">{automation.detail}</p>
      <div className="flex items-center justify-between border-t border-slate-100 pt-2 text-[11px] text-slate-500">
        <span>{automation.lastRun}</span>
        <span className="flex items-center gap-1">
          <span className={cn("h-1.5 w-1.5 rounded-full", health.dot)} />
          {health.label}
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
      className="mx-auto w-full max-w-5xl rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm backdrop-blur"
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-sky-600">
            Agent workspace
          </p>
          <h2
            id="featured-automations-heading"
            className="mt-1 text-lg font-semibold text-slate-900"
          >
            How your automations are working
          </h2>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
          Demo data
        </span>
      </div>

      <div className="flex flex-wrap gap-2" aria-label="Open automations">
        {DEMO_AUTOMATIONS.map((automation) => {
          const health = HEALTH_STYLE[automation.health];

          return (
            <StyledTooltip
              key={automation.id}
              content={<AutomationTooltip automation={automation} />}
              placement="bottom"
              tooltipClassName="!max-w-none !bg-white"
            >
              <button
                type="button"
                onClick={() => addFeatured(automation)}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-sky-300 hover:bg-sky-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <span className={cn("h-2 w-2 rounded-full", health.dot)} />
                {automation.name}
              </button>
            </StyledTooltip>
          );
        })}
      </div>

      <div className="mt-5 border-t border-slate-100 pt-4">
        <div className="mb-3 flex items-center gap-2">
          <Sparkles size={16} className="text-sky-600" aria-hidden="true" />
          <h3 className="font-medium text-slate-900">Featured automations</h3>
        </div>
        {featured.length === 0 ? (
          <p className="rounded-xl bg-slate-50 px-4 py-6 text-sm text-slate-500">
            Select an automation above to keep its latest result in view.
          </p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {featured.map((automation) => {
              const health = HEALTH_STYLE[automation.health];
              const Icon = health.icon;

              return (
                <article
                  key={automation.id}
                  className="rounded-xl border border-slate-200 bg-slate-50 p-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h4 className="font-medium text-slate-900">
                        {automation.name}
                      </h4>
                      <p className="mt-1 text-sm text-slate-600">
                        {automation.result}
                      </p>
                    </div>
                    <Icon
                      size={20}
                      className={cn(
                        automation.health === "healthy"
                          ? "text-emerald-600"
                          : automation.health === "attention"
                            ? "text-amber-500"
                            : "text-sky-500",
                      )}
                      aria-label={health.label}
                    />
                  </div>
                  <p className="mt-4 text-xs text-slate-500">
                    {automation.lastRun} · {automation.nextRun}
                  </p>
                </article>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
