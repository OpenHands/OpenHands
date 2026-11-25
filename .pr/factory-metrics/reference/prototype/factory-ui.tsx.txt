/* eslint-disable i18next/no-literal-string -- PROTOTYPE: wireframe copy, not yet localized. */
import React from "react";
import { X } from "lucide-react";
import { cn } from "#/utils/utils";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";

export type Tone = "neutral" | "good" | "warn" | "bad" | "info";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "bg-surface-raised text-muted",
  good: "bg-success-soft text-success-soft-foreground",
  warn: "bg-warning-soft text-warning-soft-foreground",
  bad: "bg-danger-soft text-danger-soft-foreground",
  info: "bg-info-soft text-info-soft-foreground",
};

export function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: Tone;
}) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded px-1.5 py-0.5 text-2xs font-medium",
        TONE_CLASS[tone],
      )}
    >
      {children}
    </span>
  );
}

export function Dot({ tone = "neutral" }: { tone?: Tone }) {
  const color: Record<Tone, string> = {
    neutral: "bg-muted",
    good: "bg-success",
    warn: "bg-warning",
    bad: "bg-danger",
    info: "bg-info",
  };
  return (
    <span className={cn("inline-block size-1.5 rounded-full", color[tone])} />
  );
}

type ButtonVariant = "primary" | "secondary" | "ghost";

export function Button({
  children,
  variant = "secondary",
  size = "md",
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: "sm" | "md";
}) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-md border font-medium transition-colors disabled:cursor-default disabled:opacity-40",
        size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-sm",
        variant === "primary" &&
          "border-transparent bg-primary text-on-accent hover:opacity-85",
        variant === "secondary" &&
          "border-border bg-canvas-base text-content hover:bg-surface-raised",
        variant === "ghost" &&
          "border-transparent text-muted hover:bg-surface-raised hover:text-content",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Card({
  children,
  className,
  onClick,
  tight,
}: {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  tight?: boolean;
}) {
  const cls = cn(
    "min-w-0 rounded-xl border border-border bg-canvas-base text-left",
    tight ? "p-3" : "p-4",
    onClick &&
      "cursor-pointer transition-colors hover:border-contrast/30 hover:bg-surface-raised/40",
    className,
  );
  if (onClick) {
    return (
      <button
        type="button"
        className={cn(cls, "block w-full")}
        onClick={onClick}
      >
        {children}
      </button>
    );
  }
  return <div className={cls}>{children}</div>;
}

export function CardTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-3 text-sm font-semibold text-content">{children}</h3>
  );
}

export function Muted({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={cn("text-xs text-muted", className)}>{children}</div>;
}

export function Stat({
  label,
  value,
  hint,
  small,
}: {
  label: string;
  value: string;
  hint?: string;
  small?: boolean;
}) {
  return (
    <Card>
      <div className="text-xs font-medium text-muted">{label}</div>
      <div
        className={cn(
          "my-1 font-semibold tracking-tight text-content",
          small ? "text-xl" : "text-3xl",
        )}
      >
        {value}
      </div>
      {hint ? <Muted>{hint}</Muted> : null}
    </Card>
  );
}

export function PageHeading({
  eyebrow,
  title,
  subtitle,
  actions,
  backTo,
  backLabel,
}: {
  eyebrow?: string;
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  backTo?: string;
  backLabel?: string;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        {backTo ? (
          <NavigationLink
            to={backTo}
            className="mb-2 inline-block text-xs text-muted hover:text-content"
          >
            ← {backLabel}
          </NavigationLink>
        ) : null}
        {eyebrow ? (
          <div className="text-2xs font-semibold uppercase tracking-wider text-muted">
            {eyebrow}
          </div>
        ) : null}
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-content">
          {title}
        </h1>
        {subtitle ? (
          <p className="mt-1 text-sm text-muted">{subtitle}</p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex flex-wrap items-center gap-2">{actions}</div>
      ) : null}
    </div>
  );
}

export function SectionTitle({
  children,
  right,
}: {
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-3 mt-7 flex items-center justify-between gap-2">
      <h2 className="text-sm font-semibold text-content">{children}</h2>
      {right}
    </div>
  );
}

export function Note({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-border bg-surface-raised/40 p-3 text-xs leading-relaxed text-muted">
      {children}
    </div>
  );
}

export function Callout({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-6 rounded border-l-2 border-info bg-info-soft/40 px-4 py-3 text-xs leading-relaxed text-muted">
      {children}
    </div>
  );
}

export function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded border border-border px-1.5 py-0.5 text-2xs text-muted">
      {children}
    </span>
  );
}

export function Bars({
  values,
  height = "h-24",
}: {
  values: number[];
  height?: string;
}) {
  return (
    <div
      className={cn("flex items-end gap-1 border-b border-border pt-2", height)}
    >
      {values.map((v, i) => (
        <span
          key={i}
          className={cn(
            "flex-1 rounded-t-sm",
            i % 3 === 2 ? "bg-primary/70" : "bg-primary/25",
          )}
          style={{ height: `${v}%` }}
        />
      ))}
    </div>
  );
}

export function Row({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick?: () => void;
}) {
  const cls =
    "flex w-full items-center justify-between gap-3 border-b border-border py-3 text-left text-sm last:border-b-0";
  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        className={cn(cls, "cursor-pointer hover:bg-surface-raised/40")}
      >
        {children}
      </button>
    );
  }
  return <div className={cls}>{children}</div>;
}

export interface Column<T> {
  header: string;
  width?: string;
  render: (row: T) => React.ReactNode;
}

export function Table<T>({
  columns,
  rows,
  onRowClick,
  rowKey,
}: {
  columns: Column<T>[];
  rows: T[];
  onRowClick?: (row: T) => void;
  rowKey: (row: T) => string;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full table-fixed border-collapse text-sm">
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.header}
                style={c.width ? { width: c.width } : undefined}
                className="border-b border-border bg-surface-raised/40 px-4 py-2.5 text-left text-2xs font-semibold uppercase tracking-wide text-muted"
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={cn(
                "border-t border-border first:border-t-0",
                onRowClick && "cursor-pointer hover:bg-surface-raised/40",
              )}
            >
              {columns.map((c) => (
                <td
                  key={c.header}
                  className="truncate px-4 py-3 align-middle text-content"
                >
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Timeline({
  events,
}: {
  events: { title: string; detail: string; done?: boolean }[];
}) {
  return (
    <ol className="ml-2 mt-2 border-l border-border pl-5">
      {events.map((e) => (
        <li key={e.title} className="relative pb-5 last:pb-0">
          <span
            className={cn(
              "absolute -left-[27px] top-1 size-2.5 rounded-full border-2 border-primary",
              e.done ? "bg-primary" : "bg-canvas-base",
            )}
          />
          <div className="text-sm font-medium text-content">{e.title}</div>
          <div className="mt-1 text-xs text-muted">{e.detail}</div>
        </li>
      ))}
    </ol>
  );
}

export function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-4">
      <div className="mb-1.5 text-xs font-semibold text-muted">{label}</div>
      <div className="rounded-md border border-border bg-canvas-base px-3 py-2 text-sm text-content">
        {children}
      </div>
    </div>
  );
}

export function CodeBox({ children }: { children: string }) {
  return (
    <pre className="overflow-auto whitespace-pre-wrap rounded-lg bg-surface-deep p-4 font-mono text-xs leading-relaxed text-content">
      {children}
    </pre>
  );
}

export function FilterChips<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "cursor-pointer rounded-md border px-2.5 py-1 text-xs transition-colors",
            value === o.value
              ? "border-contrast/30 bg-tertiary text-contrast"
              : "border-border text-muted hover:bg-surface-raised hover:text-content",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Detail drawer                                                       */
/* ------------------------------------------------------------------ */

interface DrawerContent {
  title: string;
  body: React.ReactNode;
}

interface DrawerApi {
  open: (title: string, body: React.ReactNode) => void;
  /** Placeholder for actions the wireframe does not implement. */
  demo: (what?: string) => void;
  close: () => void;
}

const DrawerContext = React.createContext<DrawerApi>({
  open: () => {},
  demo: () => {},
  close: () => {},
});

export function useFactoryDrawer() {
  return React.useContext(DrawerContext);
}

export function FactoryDrawerProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [content, setContent] = React.useState<DrawerContent | null>(null);
  const close = React.useCallback(() => setContent(null), []);
  const api = React.useMemo<DrawerApi>(
    () => ({
      open: (title, body) => setContent({ title, body }),
      demo: (what) =>
        setContent({
          title: what ?? "Example interaction",
          body: (
            <>
              <p className="text-sm leading-relaxed text-content">
                This prototype is a wireframe. This action marks where the
                product would open a creation or configuration flow.
              </p>
              <div className="mt-4">
                <Note>
                  Suggested interaction: create an item, configure an agent or
                  automation, validate inputs, then return to the originating
                  view.
                </Note>
              </div>
            </>
          ),
        }),
      close,
    }),
    [close],
  );
  useCloseOnEscape(content !== null, close);

  return (
    <DrawerContext.Provider value={api}>
      {children}
      {content ? (
        <>
          <div
            className="fixed inset-0 z-40 bg-black/30"
            onClick={close}
            aria-hidden
          />
          <aside
            role="dialog"
            aria-modal="true"
            aria-label={content.title}
            data-testid="factory-drawer"
            className="fixed inset-y-0 right-0 z-50 w-[min(420px,95vw)] overflow-auto border-l border-border bg-canvas-base p-6 shadow-2xl"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-2xs font-semibold uppercase tracking-wider text-muted">
                  Detail panel
                </div>
                <h2 className="mt-1 text-lg font-semibold text-content">
                  {content.title}
                </h2>
              </div>
              <button
                type="button"
                aria-label="Close"
                onClick={close}
                className="cursor-pointer rounded-md p-1.5 text-muted hover:bg-surface-raised hover:text-content"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="mt-4">{content.body}</div>
          </aside>
        </>
      ) : null}
    </DrawerContext.Provider>
  );
}
