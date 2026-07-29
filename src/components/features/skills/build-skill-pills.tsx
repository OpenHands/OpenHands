import React from "react";
import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";
import { cn } from "#/utils/utils";
import { SkillTypeBadge } from "./skill-type-badge";
import {
  SKILL_CARD_PILL_CLASS,
  type SkillCardPill,
} from "./skill-card-pill-row";

type SkillPillVariant = "card" | "detail";

interface BuildSkillPillsOptions {
  variant?: SkillPillVariant;
  testIdPrefix?: string;
}

function pillTestId(
  prefix: string | undefined,
  skillName: string,
  suffix: string,
) {
  if (prefix) {
    return `${prefix}-${skillName}-${suffix}`;
  }
  return undefined;
}

export function buildSkillPills(
  skill: SkillInfo,
  translate: TFunction,
  options: BuildSkillPillsOptions = {},
): SkillCardPill[] {
  const { variant = "card", testIdPrefix } = options;
  const pills: SkillCardPill[] = [
    {
      id: `type-${skill.type}`,
      node: <SkillTypeBadge type={skill.type} />,
    },
  ];

  if (skill.version) {
    pills.push({
      id: `version-${skill.version}`,
      node: (
        <span
          data-testid={
            pillTestId(testIdPrefix, skill.name, "version") ??
            `skill-version-${skill.name}`
          }
          className={SKILL_CARD_PILL_CLASS}
        >
          {translate(I18nKey.SETTINGS$SKILLS_VERSION, {
            version: skill.version,
          })}
        </span>
      ),
    });
  }

  if (variant === "detail" && skill.license) {
    pills.push({
      id: `license-${skill.license}`,
      node: (
        <span
          data-testid={pillTestId(testIdPrefix, skill.name, "license")}
          className={SKILL_CARD_PILL_CLASS}
        >
          {skill.license}
        </span>
      ),
    });
  }

  if (variant === "detail" && skill.compatibility) {
    pills.push({
      id: `compatibility-${skill.compatibility}`,
      node: (
        <span
          data-testid={pillTestId(testIdPrefix, skill.name, "compatibility")}
          className={SKILL_CARD_PILL_CLASS}
        >
          {skill.compatibility}
        </span>
      ),
    });
  }

  if (skill.disable_model_invocation) {
    pills.push({
      id: "disable-model-invocation",
      node: (
        <span
          data-testid={
            pillTestId(testIdPrefix, skill.name, "disable-model-invocation") ??
            `skill-disable-model-invocation-${skill.name}`
          }
          className="inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full border border-[rgba(248,113,113,0.35)] bg-[rgba(248,113,113,0.12)] px-2 py-0.5 text-[11px] font-medium leading-4 text-[#fca5a5]"
        >
          <span className="size-1.5 rounded-full bg-[#fca5a5]" />
          {translate(I18nKey.SETTINGS$SKILLS_DISABLE_MODEL_INVOCATION)}
        </span>
      ),
    });
  }

  if (variant === "detail" && skill.allowed_tools) {
    for (const tool of skill.allowed_tools) {
      pills.push({
        id: `allowed-tool-${tool}`,
        node: (
          <span
            data-testid={pillTestId(testIdPrefix, skill.name, `tool-${tool}`)}
            className={cn(SKILL_CARD_PILL_CLASS, "font-mono")}
          >
            {tool}
          </span>
        ),
      });
    }
  }

  if (variant === "detail" && skill.metadata) {
    for (const [key, value] of Object.entries(skill.metadata)) {
      pills.push({
        id: `metadata-${key}`,
        node: (
          <span
            data-testid={pillTestId(
              testIdPrefix,
              skill.name,
              `metadata-${key}`,
            )}
            className={SKILL_CARD_PILL_CLASS}
          >
            <span className="font-mono text-[10px] text-tertiary-light">
              {key}:
            </span>{" "}
            {value}
          </span>
        ),
      });
    }
  }

  for (const trigger of skill.triggers ?? []) {
    pills.push({
      id: `trigger-${trigger}`,
      node: (
        <span
          data-testid={pillTestId(
            testIdPrefix,
            skill.name,
            `trigger-${trigger}`,
          )}
          className={SKILL_CARD_PILL_CLASS}
        >
          {trigger}
        </span>
      ),
    });
  }

  return pills;
}
