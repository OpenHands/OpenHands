import React from "react";

interface ErrorMessageProps {
  isError: boolean;
  message?: string;
  testId?: string;
}

export function ErrorMessage({
  isError,
  message = "Failed to load data",
  testId = "dropdown-error",
}: ErrorMessageProps) {
  if (!isError) return null;

  return (
    <div className="text-red-500 text-sm mt-1" data-testid={testId}>
      {message}
    </div>
  );
}
