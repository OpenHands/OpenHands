import React from "react";

interface KeyValueRow {
  /** Already-translated label. */
  label: string;
  value: React.ReactNode;
}

/**
 * Two-column label / value grid for compact parameter displays.
 */
export function KeyValueGrid({ rows }: { rows: KeyValueRow[] }) {
  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
      {rows.map(({ label, value }) => (
        <React.Fragment key={label}>
          <span className="text-muted">{label}</span>
          <span className="break-all font-mono text-foreground">{value}</span>
        </React.Fragment>
      ))}
    </div>
  );
}
