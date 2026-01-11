import React from "react";
import { ExtraProps } from "react-markdown";

export function anchor({
  href,
  children,
}: React.ClassAttributes<HTMLAnchorElement> &
  React.AnchorHTMLAttributes<HTMLAnchorElement> &
  ExtraProps) {
  return (
    <a
      className="text-accent hover:underline"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  );
}
