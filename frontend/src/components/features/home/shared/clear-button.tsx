import React from "react";
import { cn } from "#/utils/utils";

interface ClearButtonProps {
  disabled: boolean;
  onClear: () => void;
  testId?: string;
}

export function ClearButton({
  disabled,
  onClear,
  testId = "dropdown-clear",
}: ClearButtonProps) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClear();
      }}
      disabled={disabled}
      className={cn(
        "p-1 text-[#fff]",
        "cursor-pointer disabled:cursor-not-allowed disabled:opacity-60",
      )}
      type="button"
      aria-label="Clear selection"
      data-testid={testId}
    >
      <svg
        className="w-4 h-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M6 18L18 6M6 6l12 12"
        />
      </svg>
    </button>
  );
}
