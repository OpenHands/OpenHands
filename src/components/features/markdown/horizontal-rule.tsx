import React from "react";
import { ExtraProps } from "react-markdown";

// Custom component to render <hr> in markdown
export function hr({
  ...props
}: React.ClassAttributes<HTMLHRElement> &
  React.HTMLAttributes<HTMLHRElement> &
  ExtraProps) {
  return (
    <hr
      {...props}
      className="my-4 border-0 border-t border-[var(--oh-border)]"
    />
  );
}
