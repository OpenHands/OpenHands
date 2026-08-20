import FolderIcon from "#/icons/folder.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import SparkleIcon from "#/icons/sparkle.svg?react";
import { Plug, Zap } from "lucide-react";
import { INTEGRATION_CATALOG as MCP_MARKETPLACE } from "@openhands/extensions/integrations";
import type { SkillCardPill } from "#/components/features/skills/skill-card-pill-row";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import type { Automation } from "#/types/automation";
import { cn } from "#/utils/utils";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { getMarketplaceEntryById } from "#/utils/mcp-marketplace-utils";
import {
  formatTriggerSourceLabel,
  getTriggerEventLabel,
  getTriggerSource,
} from "#/components/features/home/featured-automations/automation-run-health";

export function buildAutomationMetadataPills(
  automation: Automation,
  scheduleLabel: string,
): SkillCardPill[] {
  const pills: SkillCardPill[] = [];

  if (automation.repository) {
    pills.push({
      id: "repository",
      node: (
        <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
          <FolderIcon className="size-3 shrink-0" />
          {automation.repository}
        </span>
      ),
    });
  }

  if (automation.trigger.type === "event") {
    const eventLabel = getTriggerEventLabel(automation);
    if (eventLabel) {
      pills.push({
        id: "event-trigger",
        node: (
          <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
            <Zap className="size-3 shrink-0" aria-hidden="true" />
            {eventLabel}
          </span>
        ),
      });
    }

    const source = getTriggerSource(automation);
    if (source) {
      const sourceEntry = getMarketplaceEntryById(
        source.toLowerCase(),
        MCP_MARKETPLACE,
      );
      pills.push({
        id: "event-source",
        node: (
          <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
            {sourceEntry ? (
              <McpLogoBadge
                entry={sourceEntry}
                size="xs"
                testId="automation-source-logo"
              />
            ) : (
              <Plug
                className="size-3 shrink-0"
                aria-hidden="true"
                data-testid="automation-source-logo"
              />
            )}
            {formatTriggerSourceLabel(source)}
          </span>
        ),
      });
    }
  } else {
    pills.push({
      id: "schedule",
      node: (
        <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
          <ClockIcon className="size-3 shrink-0" />
          {scheduleLabel}
        </span>
      ),
    });
  }

  if (automation.model) {
    pills.push({
      id: "model",
      node: (
        <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
          <SparkleIcon className="size-3 shrink-0" />
          {automation.model}
        </span>
      ),
    });
  }

  return pills;
}
