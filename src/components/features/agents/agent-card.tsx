import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { AgentInfo } from "#/types/settings";
import { cn } from "#/utils/utils";
import CopyIcon from "#/icons/copy.svg?react";
import CheckmarkIcon from "#/icons/checkmark.svg?react";
import { CirclePlusCheckToggle } from "#/components/shared/buttons/circle-plus-check-toggle";
import { SkillCardPillRow } from "#/components/features/skills/skill-card-pill-row";
import { isCopyableSkillSource } from "#/components/features/skills/is-copyable-skill-source";
import {
  extensionModuleCardInteractiveClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";
import { AgentIconBadge } from "./agent-icon-badge";
import { getAgentCardDescription } from "./get-agent-card-description";
import { buildAgentPills } from "./build-agent-pills";

interface AgentCardProps {
  agent: AgentInfo;
  enabled: boolean;
  onOpen: () => void;
  onToggle: (enabled: boolean) => void;
}

export function AgentCard({
  agent,
  enabled,
  onOpen,
  onToggle,
}: AgentCardProps) {
  const { t } = useTranslation("openhands");
  const [sourceCopied, setSourceCopied] = React.useState(false);

  const description = getAgentCardDescription(agent);
  const pills = React.useMemo(() => buildAgentPills(agent), [agent]);
  const showCopySource = isCopyableSkillSource(agent.source);

  const handleCopySource = async (event: React.MouseEvent) => {
    event.stopPropagation();
    if (!agent.source) {
      return;
    }

    await navigator.clipboard.writeText(agent.source);
    setSourceCopied(true);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpen();
    }
  };

  React.useEffect(() => {
    if (!sourceCopied) {
      return undefined;
    }

    const timeout = setTimeout(() => setSourceCopied(false), 2000);
    return () => clearTimeout(timeout);
  }, [sourceCopied]);

  return (
    <div
      data-testid={`agent-card-${agent.name}`}
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={handleKeyDown}
      className={cn(
        "flex min-w-0 flex-col gap-3 overflow-hidden p-4",
        extensionModuleCardSurfaceClassName,
        extensionModuleCardInteractiveClassName,
      )}
    >
      <div className="flex items-start gap-3">
        <AgentIconBadge agentName={agent.name} />
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <header className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <h3
                data-testid={`agent-name-${agent.name}`}
                className="truncate text-sm font-semibold text-white"
              >
                {agent.name}
              </h3>
              {agent.source ? (
                <div className="mt-0.5 flex min-w-0 items-center gap-1">
                  <p
                    data-testid={`agent-source-${agent.name}`}
                    className="min-w-0 flex-1 truncate text-xs text-tertiary-alt"
                    title={agent.source}
                  >
                    {agent.source}
                  </p>
                  {showCopySource ? (
                    <button
                      type="button"
                      data-testid={`agent-copy-source-${agent.name}`}
                      aria-label={t(
                        sourceCopied
                          ? I18nKey.BUTTON$COPIED
                          : I18nKey.SETTINGS$AGENTS_COPY_PATH,
                      )}
                      disabled={sourceCopied}
                      onClick={handleCopySource}
                      className="shrink-0 cursor-pointer border-0 bg-transparent p-0.5 text-tertiary-alt hover:text-white disabled:cursor-default [&_path]:fill-current"
                    >
                      {sourceCopied ? (
                        <CheckmarkIcon width={12} height={12} />
                      ) : (
                        <CopyIcon width={12} height={12} />
                      )}
                    </button>
                  ) : null}
                </div>
              ) : null}
            </div>
            <CirclePlusCheckToggle
              testId={`agent-toggle-${agent.name}`}
              isSelected={enabled}
              onToggle={onToggle}
              disableTooltipKey={I18nKey.COMMON$DISABLE}
            />
          </header>

          {description ? (
            <div
              data-testid={`agent-description-${agent.name}`}
              className="min-w-0"
            >
              <p className="line-clamp-2 break-words text-xs leading-relaxed text-tertiary-light">
                {description}
              </p>
            </div>
          ) : null}

          {pills.length > 0 ? (
            <SkillCardPillRow
              pills={pills}
              testId={`agent-pills-${agent.name}`}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}
