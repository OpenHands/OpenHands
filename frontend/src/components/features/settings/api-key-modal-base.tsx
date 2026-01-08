import React, { ReactNode } from "react";

interface ApiKeyModalBaseProps {
  title: string;
  width?: string;
  children: ReactNode;
  footer: ReactNode;
}

/**
 * Shared wrapper for API key modals.
 * Provides consistent styling for all API key modal content.
 * Visibility is managed by the centralized modal system.
 */
export function ApiKeyModalBase({
  title,
  width = "500px",
  children,
  footer,
}: ApiKeyModalBaseProps) {
  return (
    <div
      className="bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-tertiary"
      style={{ width }}
    >
      <h3 className="text-xl font-bold">{title}</h3>
      {children}
      <div className="w-full flex gap-2 mt-2">{footer}</div>
    </div>
  );
}
