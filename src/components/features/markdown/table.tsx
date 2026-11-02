import React from "react";
import { ExtraProps } from "react-markdown";
import { MarkdownTableScroll } from "./markdown-table-scroll";

// Cells wrap once they reach this width so one long sentence cannot stretch the
// table far past the viewport. A fixed cap (rather than a share of the chat
// width) avoids container-query/measurement feedback loops inside shrink-to-fit
// parents such as the user message bubble.
const CELL_WIDTH_CLASS = "max-w-md break-words";

// Custom component to render <table> in markdown
export function table({
  children,
}: React.ClassAttributes<HTMLTableElement> &
  React.TableHTMLAttributes<HTMLTableElement> &
  ExtraProps) {
  return (
    <MarkdownTableScroll>
      <table
        className={[
          "my-4 w-max min-w-full border-separate border-spacing-0 overflow-hidden rounded-xl border border-border text-sm",
          "[&_td]:border-b [&_td]:border-r [&_th]:border-b [&_th]:border-r",
          "[&_td:last-child]:border-r-0 [&_th:last-child]:border-r-0",
          "[&_tbody_tr:last-child_td]:border-b-0",
        ].join(" ")}
      >
        {children}
      </table>
    </MarkdownTableScroll>
  );
}

// Custom component to render <th> in markdown
export function th({
  children,
}: React.ClassAttributes<HTMLTableCellElement> &
  React.ThHTMLAttributes<HTMLTableCellElement> &
  ExtraProps) {
  return (
    <th
      className={`${CELL_WIDTH_CLASS} border-border bg-surface px-3 py-2 text-left font-semibold text-contrast`}
    >
      {children}
    </th>
  );
}

// Custom component to render <td> in markdown
export function td({
  children,
}: React.ClassAttributes<HTMLTableCellElement> &
  React.TdHTMLAttributes<HTMLTableCellElement> &
  ExtraProps) {
  return (
    <td className={`${CELL_WIDTH_CLASS} border-border px-3 py-2 align-top`}>
      {children}
    </td>
  );
}
