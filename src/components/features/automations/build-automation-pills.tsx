import FolderIcon from "#/icons/folder.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import SparkleIcon from "#/icons/sparkle.svg?react";
import GlobeIcon from "#/icons/globe.svg?react";
import type { SkillCardPill } from "#/components/features/skills/skill-card-pill-row";
import type { Automation } from "#/types/automation";
import { cn } from "#/utils/utils";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { formatEventOn } from "#/utils/automation-schedule";

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
    const eventLabel = [
      automation.trigger.on ? formatEventOn(automation.trigger.on) : "",
      automation.trigger.source ? `(${automation.trigger.source})` : "",
    ]
      .filter(Boolean)
      .join(" ");

    pills.push({
      id: "event-trigger",
      node: (
        <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
          <GlobeIcon className="size-3 shrink-0" />
          {eventLabel}
        </span>
      ),
    });
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
