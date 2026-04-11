import React from "react";
import { ExtraProps } from "react-markdown";

// Custom component to render <table> in markdown
export function table({
  children,
}: React.ClassAttributes<HTMLTableElement> &
  React.TableHTMLAttributes<HTMLTableElement> &
  ExtraProps) {
  return (
    <div className="my-4 w-full overflow-x-auto">
      <table className="w-full border-collapse border border-neutral-600 text-sm">
        {children}
      </table>
    </div>
  );
}

// Custom component to render <th> in markdown
export function th({
  children,
}: React.ClassAttributes<HTMLTableCellElement> &
  React.ThHTMLAttributes<HTMLTableCellElement> &
  ExtraProps) {
  return (
    <th className="border border-neutral-600 bg-neutral-800 px-3 py-2 text-left font-semibold text-white">
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
    <td className="border border-neutral-600 px-3 py-2 align-top">
      {children}
    </td>
  );
}
