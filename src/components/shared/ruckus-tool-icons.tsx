import type { ReactNode } from "react";

function ToolIcon({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinejoin="miter"
      className={className}
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

export function RuckusFilesIcon({ className }: { className: string }) {
  return (
    <ToolIcon className={className}>
      <path d="M5 3h10l4 4v14H5V3Z" fill="currentColor" fillOpacity=".14" />
      <path d="M14 3v5h5M8 12h8M8 16h6" />
    </ToolIcon>
  );
}
export function RuckusCommitsIcon({ className }: { className: string }) {
  return (
    <ToolIcon className={className}>
      <path d="M5 3h10l4 4v14H5V3Z" fill="currentColor" fillOpacity=".14" />
      <path d="M14 3v5h5M8 12h8M8 17h6M11 9v6" />
    </ToolIcon>
  );
}
export function RuckusTerminalIcon({ className }: { className: string }) {
  return (
    <ToolIcon className={className}>
      <path d="M3 4h18v16H3Z" fill="currentColor" fillOpacity=".14" />
      <path d="m7 8 4 4-4 4m6 0h4" />
    </ToolIcon>
  );
}
export function RuckusBrowserIcon({ className }: { className: string }) {
  return (
    <ToolIcon className={className}>
      <circle cx="12" cy="12" r="9" fill="currentColor" fillOpacity=".14" />
      <ellipse cx="12" cy="12" rx="4" ry="9" />
      <path d="M3 12h18M5 7h14M5 17h14" />
    </ToolIcon>
  );
}
export function RuckusPlannerIcon({ className }: { className: string }) {
  return (
    <ToolIcon className={className}>
      <path
        d="M3 3h6v6H3zm12 12h6v6h-6Z"
        fill="currentColor"
        fillOpacity=".2"
      />
      <path d="M15 3h6v6h-6ZM3 18h6m-3-3v6m3-15h6m3 3v6" />
    </ToolIcon>
  );
}
export function RuckusTasksIcon({ className }: { className: string }) {
  return (
    <ToolIcon className={className}>
      <path d="M3 3h18v18H3Z" fill="currentColor" fillOpacity=".14" />
      <path d="m6 8 2 2 3-4m2 2h5M6 16h3m4 0h5" />
    </ToolIcon>
  );
}
export function RuckusUsageIcon({ className }: { className: string }) {
  return (
    <ToolIcon className={className}>
      <path
        d="M4 13h4v7H4zm6-9h4v16h-4zm6 5h4v11h-4Z"
        fill="currentColor"
        fillOpacity=".2"
      />
    </ToolIcon>
  );
}
