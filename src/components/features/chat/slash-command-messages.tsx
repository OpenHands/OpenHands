import { useTranslation } from "react-i18next";
import { GenericEventMessage } from "./generic-event-message";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import type { LoadedResources, SlashCommandItem } from "#/types/slash-command";
import type { SlashCommandOutput } from "#/stores/slash-command-output-store";
import { BUILT_IN_COMMANDS, CLI_HELP_COMMAND_ORDER } from "#/utils/constants";
import { getSlashCommandDescription } from "#/utils/slash-command-description";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

interface SlashCommandMessagesProps {
  outputScopeId: string | null | undefined;
  timelineBoundaryEventId?: string | null;
  outputs?: SlashCommandOutput[];
}

const CLI_HELP_COMMANDS = new Set<string>(CLI_HELP_COMMAND_ORDER);
const BUILT_IN_COMMAND_NAMES = new Set(
  BUILT_IN_COMMANDS.map((item) => item.command),
);

function HelpCommandGroup({
  title,
  commands,
  showDescriptions = true,
  collapsible = false,
}: {
  title: string;
  commands: SlashCommandItem[];
  showDescriptions?: boolean;
  collapsible?: boolean;
}) {
  const { t } = useTranslation("openhands");
  if (commands.length === 0) return null;

  const commandList = (
    <ul
      className={
        showDescriptions
          ? "flex flex-col gap-2"
          : "mt-1 grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-2"
      }
    >
      {commands.map((item) => {
        const description = getSlashCommandDescription(item, t);
        const commandLabel = (
          <span
            className={
              showDescriptions || !description
                ? "font-mono text-neutral-200"
                : "cursor-help border-b border-dotted border-neutral-500 font-mono text-neutral-200"
            }
            tabIndex={!showDescriptions && description ? 0 : undefined}
            aria-label={
              !showDescriptions && description
                ? `${item.command}: ${description}`
                : undefined
            }
          >
            {item.command}
          </span>
        );

        return (
          <li key={item.command}>
            {!showDescriptions && description ? (
              <StyledTooltip
                content={description}
                placement="top"
                shouldFlip
                tooltipClassName="max-w-80 whitespace-normal text-left"
              >
                {commandLabel}
              </StyledTooltip>
            ) : (
              commandLabel
            )}
            {showDescriptions && description ? (
              <p className="text-sm text-neutral-400">{description}</p>
            ) : null}
          </li>
        );
      })}
    </ul>
  );

  return collapsible ? (
    <details data-testid="slash-command-help-skill-commands">
      <summary className="cursor-pointer text-xs font-medium text-neutral-400">
        {title}
      </summary>
      {commandList}
    </details>
  ) : (
    <section className="flex flex-col gap-1">
      <h4 className="text-xs font-medium text-neutral-400">{title}</h4>
      {commandList}
    </section>
  );
}

function HelpCommandDetails({
  commands,
  isLoading,
}: {
  commands: SlashCommandItem[];
  isLoading: boolean;
}) {
  const { t } = useTranslation("openhands");
  const cliCommands = commands.filter((item) =>
    CLI_HELP_COMMANDS.has(item.command),
  );
  const canvasCommands = commands.filter(
    (item) =>
      BUILT_IN_COMMAND_NAMES.has(item.command) &&
      !CLI_HELP_COMMANDS.has(item.command),
  );
  const skillCommands = commands.filter(
    (item) => !BUILT_IN_COMMAND_NAMES.has(item.command),
  );

  return (
    <div
      data-testid="slash-command-help-list"
      className="custom-scrollbar-always flex max-h-[50vh] flex-col gap-3 overflow-y-auto px-2 py-1"
    >
      <HelpCommandGroup
        title={t(I18nKey.SLASH_COMMAND$CLI_COMMANDS)}
        commands={cliCommands}
      />
      <HelpCommandGroup
        title={t(I18nKey.SLASH_COMMAND$CANVAS_COMMANDS)}
        commands={canvasCommands}
      />
      <HelpCommandGroup
        title={t(I18nKey.SLASH_COMMAND$SKILL_COMMANDS, {
          count: skillCommands.length,
        })}
        commands={skillCommands}
        showDescriptions={false}
        collapsible
      />
      {isLoading ? (
        <div
          role="status"
          className="flex items-center gap-2 text-sm text-neutral-400"
          data-testid="slash-command-help-loading"
        >
          <LoadingSpinner size="small" />
          <span>{t(I18nKey.SLASH_COMMAND$LOADING_RESOURCES)}</span>
        </div>
      ) : null}
      <p className="text-sm text-neutral-400">
        {t(I18nKey.SLASH_COMMAND$AUTOCOMPLETE_TIP)}
      </p>
    </div>
  );
}

function LoadedResourcesDetails({ resources }: { resources: LoadedResources }) {
  const { t } = useTranslation("openhands");
  const hookCount =
    resources.hooks?.reduce((count, hook) => count + hook.commands.length, 0) ??
    0;
  const hasUnavailableCategories =
    resources.hooks === null || resources.mcps === null;
  const summary = [
    resources.skills.length > 0 || hasUnavailableCategories
      ? t(
          resources.skills.length === 1
            ? I18nKey.SLASH_COMMAND$SKILL_COUNT_one
            : I18nKey.SLASH_COMMAND$SKILL_COUNT_other,
          {
            count: resources.skills.length,
          },
        )
      : null,
    resources.hooks === null
      ? t(I18nKey.SLASH_COMMAND$HOOKS_UNAVAILABLE)
      : hookCount > 0
        ? t(
            hookCount === 1
              ? I18nKey.SLASH_COMMAND$HOOK_COUNT_one
              : I18nKey.SLASH_COMMAND$HOOK_COUNT_other,
            { count: hookCount },
          )
        : null,
    resources.mcps === null
      ? t(I18nKey.SLASH_COMMAND$MCPS_UNAVAILABLE)
      : resources.mcps.length > 0
        ? t(
            resources.mcps.length === 1
              ? I18nKey.SLASH_COMMAND$MCP_COUNT_one
              : I18nKey.SLASH_COMMAND$MCP_COUNT_other,
            {
              count: resources.mcps.length,
            },
          )
        : null,
  ].filter((part): part is string => part !== null);
  const hasLoadedResources =
    resources.skills.length > 0 ||
    hookCount > 0 ||
    (resources.mcps?.length ?? 0) > 0;

  return (
    <div
      data-testid="slash-command-skills-list"
      className="custom-scrollbar-always flex max-h-[50vh] flex-col gap-3 overflow-y-auto px-2 py-1 text-sm text-neutral-300"
    >
      <p>
        <span className="text-neutral-400">
          {t(I18nKey.SLASH_COMMAND$SUMMARY)}:
        </span>{" "}
        {summary.length > 0
          ? summary.join(", ")
          : t(I18nKey.SLASH_COMMAND$NO_RESOURCES_SUMMARY)}
      </p>

      {!hasLoadedResources && !hasUnavailableCategories ? (
        <p className="text-neutral-400">
          {t(I18nKey.SLASH_COMMAND$NO_LOADED_RESOURCES)}
        </p>
      ) : null}

      {resources.skills.length > 0 ? (
        <section className="flex flex-col gap-1">
          <h4 className="font-medium text-neutral-300">
            {t(I18nKey.SLASH_COMMAND$SKILLS_SECTION, {
              count: resources.skills.length,
            })}
          </h4>
          <ul className="flex flex-col gap-2">
            {resources.skills.map((skill, index) => (
              <li key={`${skill.source ?? "unknown"}-${skill.name}-${index}`}>
                <span className="font-mono text-neutral-200">
                  • {skill.name}
                </span>
                {skill.description ? (
                  <p className="pl-4 text-neutral-400">{skill.description}</p>
                ) : null}
                {skill.source ? (
                  <p className="pl-4 text-neutral-400">({skill.source})</p>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {resources.hooks && resources.hooks.length > 0 ? (
        <section className="flex flex-col gap-1">
          <h4 className="font-medium text-neutral-300">
            {t(I18nKey.SLASH_COMMAND$HOOKS_SECTION, { count: hookCount })}
          </h4>
          <ul>
            {resources.hooks.map((hook) => (
              <li key={hook.hookType}>
                • {hook.hookType}: {hook.commands.join(", ")}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {resources.mcps && resources.mcps.length > 0 ? (
        <section className="flex flex-col gap-1">
          <h4 className="font-medium text-neutral-300">
            {t(I18nKey.SLASH_COMMAND$MCPS_SECTION, {
              count: resources.mcps.length,
            })}
          </h4>
          <ul className="flex flex-col gap-2">
            {resources.mcps.map((mcp) => (
              <li key={mcp.name}>
                <span>• {mcp.name}</span>
                {mcp.transport ? (
                  <p className="pl-4 text-neutral-400">({mcp.transport})</p>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

export function SlashCommandMessages({
  outputScopeId,
  timelineBoundaryEventId,
  outputs,
}: SlashCommandMessagesProps) {
  const { t } = useTranslation("openhands");
  const entriesByScope = useSlashCommandOutputStore(
    (state) => state.entriesByScope,
  );
  const entries =
    outputs ??
    (outputScopeId
      ? (entriesByScope[outputScopeId] ?? []).filter(
          (entry) => entry.timelineBoundaryEventId === timelineBoundaryEventId,
        )
      : []);

  if (entries.length === 0) return null;

  return (
    <div data-testid="slash-command-messages" className="flex w-full flex-col">
      {entries.map((entry) =>
        entry.kind === "skills" ? (
          <div
            key={entry.id}
            data-testid={`slash-command-skills-${entry.id}`}
            data-status={entry.status}
          >
            {entry.status === "loading" ? (
              <GenericEventMessage
                title={t(I18nKey.SLASH_COMMAND$LOADED_RESOURCES)}
                details={
                  <div
                    role="status"
                    className="flex items-center gap-2 px-2 py-1 text-sm text-neutral-400"
                    data-testid="slash-command-skills-loading"
                  >
                    <LoadingSpinner size="small" />
                    <span>{t(I18nKey.SLASH_COMMAND$LOADING_RESOURCES)}</span>
                  </div>
                }
                initiallyExpanded
              />
            ) : entry.status === "ready" ? (
              <GenericEventMessage
                title={t(I18nKey.SLASH_COMMAND$LOADED_RESOURCES)}
                details={<LoadedResourcesDetails resources={entry.resources} />}
                initiallyExpanded
              />
            ) : (
              <GenericEventMessage
                title={
                  entry.errorKind === "timeout"
                    ? t(I18nKey.SLASH_COMMAND$RESOURCES_TIMEOUT)
                    : t(I18nKey.SLASH_COMMAND$RESOURCES_FAILED)
                }
                details={
                  <p
                    className="px-2 py-1 text-sm text-neutral-400"
                    data-testid="slash-command-skills-error"
                  >
                    {t(I18nKey.SLASH_COMMAND$SKILLS_RETRY)}
                  </p>
                }
                success={entry.errorKind === "timeout" ? "timeout" : "error"}
                initiallyExpanded
              />
            )}
          </div>
        ) : (
          <div
            key={entry.id}
            data-testid={`slash-command-help-${entry.id}`}
            data-status={entry.status}
          >
            <GenericEventMessage
              title={t(I18nKey.SLASH_COMMAND$AVAILABLE_COMMANDS, {
                count: entry.commands.length,
              })}
              details={
                <HelpCommandDetails
                  commands={entry.commands}
                  isLoading={entry.status === "loading"}
                />
              }
              initiallyExpanded
            />
          </div>
        ),
      )}
    </div>
  );
}
