import React from "react";
import { ExtraProps } from "react-markdown";
import {
  FaCircleInfo,
  FaLightbulb,
  FaCircleExclamation,
  FaTriangleExclamation,
  FaCircleStop,
} from "react-icons/fa6";
import type { IconType } from "react-icons";
import type { AlertType } from "./remark-github-alerts";
import { cn } from "#/utils/utils";

interface AlertConfig {
  label: string;
  icon: IconType;
  // Per-alert color overrides. We stick to Tailwind palette utilities so
  // the classes show up in the source for the v4 content scanner.
  containerClass: string;
  titleClass: string;
  iconClass: string;
}

const ALERT_CONFIG: Record<AlertType, AlertConfig> = {
  note: {
    label: "Note",
    icon: FaCircleInfo,
    containerClass: "border-l-blue-500 bg-blue-500/10",
    titleClass: "text-blue-300",
    iconClass: "text-blue-400",
  },
  tip: {
    label: "Tip",
    icon: FaLightbulb,
    containerClass: "border-l-emerald-500 bg-emerald-500/10",
    titleClass: "text-emerald-300",
    iconClass: "text-emerald-400",
  },
  important: {
    label: "Important",
    icon: FaCircleExclamation,
    containerClass: "border-l-purple-500 bg-purple-500/10",
    titleClass: "text-purple-300",
    iconClass: "text-purple-400",
  },
  warning: {
    label: "Warning",
    icon: FaTriangleExclamation,
    containerClass: "border-l-yellow-500 bg-yellow-500/10",
    titleClass: "text-yellow-300",
    iconClass: "text-yellow-400",
  },
  caution: {
    label: "Caution",
    icon: FaCircleStop,
    containerClass: "border-l-rose-500 bg-rose-500/10",
    titleClass: "text-rose-300",
    iconClass: "text-rose-400",
  },
};

// Match `markdown-alert-<type>` anywhere in the className string. The
// remark plugin emits the classes as an array which React serialises to
// a space-separated string before this component sees it.
const ALERT_CLASS_REGEX =
  /(?:^|\s)markdown-alert-(note|tip|important|warning|caution)(?:\s|$)/i;

function detectAlertType(className: string | undefined): AlertType | null {
  if (!className) return null;
  const match = className.match(ALERT_CLASS_REGEX);
  return match ? (match[1].toLowerCase() as AlertType) : null;
}

// Custom component to render <blockquote> in markdown. Standard
// blockquotes get a muted left-border treatment; GitHub-style alert
// blockquotes (tagged by the `remark-github-alerts` plugin via
// `markdown-alert markdown-alert-<type>` classes on the node) get a
// coloured panel with an icon + title row matching GitHub's rendering.
export function blockquote({
  children,
  className,
}: React.ClassAttributes<HTMLQuoteElement> &
  React.BlockquoteHTMLAttributes<HTMLQuoteElement> &
  ExtraProps) {
  const alertType = detectAlertType(className);

  if (alertType) {
    const config = ALERT_CONFIG[alertType];
    const Icon = config.icon;
    return (
      <div
        data-testid={`markdown-alert-${alertType}`}
        className={cn(
          "my-3 rounded-r-sm border-l-4 px-3 py-2",
          config.containerClass,
        )}
      >
        <p
          className={cn(
            "flex items-center gap-2 font-semibold",
            config.titleClass,
          )}
        >
          <Icon aria-hidden className={cn("shrink-0", config.iconClass)} />
          <span>{config.label}</span>
        </p>
        <div className="text-content">{children}</div>
      </div>
    );
  }

  return (
    <blockquote className="my-2 border-l-4 border-border pl-3 italic text-content-muted">
      {children}
    </blockquote>
  );
}
