/* eslint-disable i18next/no-literal-string -- PROTOTYPE: wireframe copy, not yet localized. */
import React from "react";
import { useNavigation } from "#/context/navigation-context";
import {
  THROUGHPUT,
  WORK_LOG,
  factoryAgentPath,
  factoryPath,
  factoryRefPath,
  findAgent,
  findRef,
  type AttentionItem,
  type Factory,
  type FactorySection,
} from "../factory-data";
import {
  Badge,
  Button,
  Card,
  CardTitle,
  Muted,
  PageHeading,
  useFactoryDrawer,
} from "../factory-ui";
import { LogEntryRow } from "./stream-pages";
import {
  BlockerPatternsCard,
  EscalationsCard,
  FlowStats,
  NewRuleButton,
} from "./operations";

export interface FactoryPageProps {
  factory: Factory;
}

export const eyebrow = (f: Factory) => `${f.name} / Factory`;

export function useOpenAttention(factory: Factory) {
  const { open, demo, close } = useFactoryDrawer();
  const { navigate } = useNavigation();
  return (a: AttentionItem) =>
    open(
      a.title,
      <>
        <div className="flex gap-1.5">
          <Badge tone="warn">{a.kind}</Badge>
          <Badge>{findRef(a.refId)?.label ?? a.refId}</Badge>
        </div>
        <p className="mt-4 text-sm leading-relaxed text-content">{a.detail}</p>
        <p className="mt-3 text-sm text-muted">
          This work is paused for human input. Review the context, submit a
          decision, and the factory picks the work back up.
        </p>
        {a.kind === "Question" ? (
          <textarea
            className="mt-4 min-h-24 w-full rounded-md border border-border bg-canvas-base p-2 text-sm text-content"
            placeholder="Type your answer…"
          />
        ) : null}
        <div className="my-4 h-px bg-border" />
        <div className="flex flex-wrap gap-2">
          <Button
            onClick={() => {
              close();
              navigate(factoryRefPath(factory.id, a.refId));
            }}
          >
            Open {findRef(a.refId)?.type === "pr" ? "PR" : "issue"}
          </Button>
          <div className="flex-1" />
          <Button onClick={() => demo("Request changes")}>
            Request changes
          </Button>
          <Button variant="primary" onClick={() => demo("Approve")}>
            {a.kind === "Question" ? "Send answer" : "Approve"}
          </Button>
        </div>
      </>,
    );
}

/** Tiny trend line for a KPI. */
function Sparkline({ values }: { values: number[] }) {
  const w = 80;
  const h = 26;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const pts = values
    .map(
      (v, i) =>
        `${(i / (values.length - 1)) * w},${h - ((v - min) / (max - min || 1)) * (h - 4) - 2}`,
    )
    .join(" ");
  return (
    <svg width={w} height={h} className="shrink-0 text-primary" aria-hidden>
      <polyline
        points={pts}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  );
}

/**
 * The factory's home page and operator view in one: is it reaching its
 * goal, is work moving, what needs a person, and why work gets stuck.
 */
export function OverviewPage({ factory }: FactoryPageProps) {
  const { navigate } = useNavigation();
  const { demo } = useFactoryDrawer();
  const orchestrator = findAgent(factory.orchestratorId);
  const go = (section: FactorySection) =>
    navigate(factoryPath(factory.id, section));

  return (
    <>
      <PageHeading
        eyebrow={eyebrow(factory)}
        title={factory.name}
        subtitle={factory.description || "Factory overview"}
        actions={
          <>
            <Button onClick={() => demo("Edit goal and KPIs")}>
              Edit goal & KPIs
            </Button>
            <Button variant="primary" onClick={() => go("stream")}>
              Open work stream
            </Button>
          </>
        }
      />

      {/* 1. Is it reaching its goal? */}
      <Card className="p-4">
        <div className="grid gap-4 xl:grid-cols-5">
          <div className="min-w-0 xl:col-span-2">
            <div className="text-2xs font-semibold uppercase tracking-wider text-muted">
              Goal
            </div>
            <p className="mt-1 text-sm leading-relaxed text-content">
              {factory.goal || "No goal set yet."}
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {factory.owner ? <Badge>Owner · {factory.owner}</Badge> : null}
              <Badge>{factory.tag}</Badge>
            </div>
          </div>
          {factory.kpis.length ? (
            <div className="grid gap-2 sm:grid-cols-3 xl:col-span-3">
              {factory.kpis.map((k, i) => (
                <div
                  key={k.label}
                  className="flex min-w-0 flex-col rounded-lg border border-border px-3 py-2"
                >
                  <Muted className="truncate">{k.label}</Muted>
                  <div className="mt-0.5 flex min-w-0 items-center justify-between gap-2">
                    <span className="text-xl font-semibold text-content">
                      {k.value}
                    </span>
                    {i === 0 ? <Sparkline values={THROUGHPUT} /> : null}
                  </div>
                  <div className="mt-1">
                    <Badge tone={k.onTrack ? "good" : "warn"}>
                      target {k.target}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <button
              type="button"
              onClick={() => demo("Add a KPI")}
              className="cursor-pointer rounded-lg border border-dashed border-border px-3 py-4 text-left text-sm text-muted hover:text-content xl:col-span-3"
            >
              + Add a KPI to measure this factory against its goal
            </button>
          )}
        </div>
      </Card>

      {/* 2. Is work moving? */}
      <div className="mt-4">
        <FlowStats />
      </div>

      <div className="mt-4 flex flex-col gap-4 xl:flex-row">
        <div className="min-w-0 flex-1">
          {/* 3. What needs me? */}
          <EscalationsCard factory={factory} />

          {/* 4. Why does it get stuck? */}
          <div className="mt-4">
            <BlockerPatternsCard
              actions={
                <>
                  <NewRuleButton size="sm" />
                  <Button size="sm" onClick={() => go("workflows")}>
                    All rules →
                  </Button>
                </>
              }
            />
          </div>
        </div>

        {/* 5. Side column */}
        <div className="grid w-full shrink-0 content-start items-start gap-4 md:grid-cols-2 xl:w-80 xl:grid-cols-1">
          <Card>
            <CardTitle>Primary agent</CardTitle>
            {orchestrator ? (
              <>
                <button
                  type="button"
                  onClick={() =>
                    navigate(factoryAgentPath(factory.id, orchestrator.id))
                  }
                  className="flex w-full cursor-pointer items-center gap-3 text-left"
                >
                  <div className="grid size-9 place-items-center rounded-lg border border-border bg-surface-raised text-xs font-bold text-muted">
                    FC
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-content">
                      {orchestrator.name}
                    </div>
                    <Muted>acts as {orchestrator.runsAs}</Muted>
                  </div>
                </button>
                <p className="mt-3 text-sm text-muted">
                  Watching 5 blocked items · applied 3 rules today · next
                  summary at 17:00
                </p>
              </>
            ) : (
              <>
                <Muted>
                  No orchestrator. This factory is driven by its automations.
                </Muted>
                <div className="mt-3">
                  <Button size="sm" onClick={() => go("agents")}>
                    Assign a primary agent
                  </Button>
                </div>
              </>
            )}
          </Card>
          <Card>
            <div className="mb-1 flex items-center">
              <h3 className="text-sm font-semibold text-content">
                Recent activity
              </h3>
              <div className="flex-1" />
              <Button size="sm" variant="ghost" onClick={() => go("stream")}>
                All →
              </Button>
            </div>
            {WORK_LOG.slice(0, 5).map((e) => (
              <LogEntryRow key={e.id} entry={e} factoryId={factory.id} />
            ))}
          </Card>
        </div>
      </div>
    </>
  );
}
