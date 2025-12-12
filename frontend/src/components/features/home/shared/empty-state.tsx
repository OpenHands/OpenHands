import React from "react";

interface EmptyStateProps {
  inputValue: string;
  searchMessage?: string;
  emptyMessage?: string;
  testId?: string;
}

export function EmptyState({
  inputValue,
  searchMessage = "No items found",
  emptyMessage = "No items available",
  testId = "dropdown-empty",
}: EmptyStateProps) {
  return (
    <li
      className="px-3 py-2 text-[#B7BDC2] text-sm rounded-lg mx-0.5 my-0.5"
      data-testid={testId}
    >
      {inputValue ? searchMessage : emptyMessage}
    </li>
  );
}
