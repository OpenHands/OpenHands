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
  // Literal semantic recipes remain visible to the Tailwind content scanner.
  containerClass: string;
}

const ALERT_CONFIG: Record<AlertType, AlertConfig> = {
  note: {
    label: "Note",
    icon: FaCircleInfo,
    containerClass:
      "border-l-markdown-alert-note bg-markdown-alert-note/5 text-markdown-alert-note",
  },
  tip: {
    label: "Tip",
    icon: FaLightbulb,
    containerClass:
      "border-l-markdown-alert-tip bg-markdown-alert-tip/5 text-markdown-alert-tip",
  },
  important: {
    label: "Important",
    icon: FaCircleExclamation,
    containerClass:
      "border-l-markdown-alert-important bg-markdown-alert-important/5 text-markdown-alert-important",
  },
  warning: {
    label: "Warning",
    icon: FaTriangleExclamation,
    containerClass:
      "border-l-markdown-alert-warning bg-markdown-alert-warning/5 text-markdown-alert-warning",
  },
  caution: {
    label: "Caution",
    icon: FaCircleStop,
    containerClass:
      "border-l-markdown-alert-caution bg-markdown-alert-caution/5 text-markdown-alert-caution",
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
        <p className="flex items-center gap-2 font-semibold">
          <Icon aria-hidden className="shrink-0" />
          <span>{config.label}</span>
        </p>
        <div className="text-contrast prose-strong:text-inherit prose-headings:text-contrast prose-code:text-contrast prose-kbd:text-contrast prose-figcaption:text-contrast marker:text-contrast">
          {children}
        </div>
      </div>
    );
  }

  return (
    <blockquote className="my-2 border-l-4 border-border pl-3 italic text-content-muted">
      {children}
    </blockquote>
  );
}
