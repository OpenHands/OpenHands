import { useTranslation } from "react-i18next";
import { GenericEventMessage } from "./generic-event-message";
import { I18nKey } from "#/i18n/declaration";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { getSkillDescription } from "#/utils/slash-command-description";
import { getInstalledServerTitle } from "#/utils/mcp-installed-server-display";
import { SKILLS_COMMAND } from "#/utils/constants";
import { SkillIconBadge } from "#/components/features/skills/skill-icon-badge";
import { SkillCardPillRow } from "#/components/features/skills/skill-card-pill-row";
import { buildSkillPills } from "#/components/features/skills/build-skill-pills";
import { getSkillCardDescription } from "#/components/features/skills/get-skill-card-description";
import { extensionModuleCardSurfaceClassName } from "#/utils/extension-module-card-classes";

interface SlashCommandOutputMessagesProps {
  conversationId: string | null | undefined;
  anchorEventId: string | null;
}

export function SlashCommandOutputMessages({
  conversationId,
  anchorEventId,
}: SlashCommandOutputMessagesProps) {
  const { t } = useTranslation("openhands");
  const entriesByConversation = useSlashCommandOutputStore(
    (state) => state.entriesByConversation,
  );
  const entries = conversationId
    ? (entriesByConversation[conversationId] ?? []).filter(
        (entry) => entry.anchorEventId === anchorEventId,
      )
    : [];

  if (!conversationId || entries.length === 0) return null;

  return (
    <div
      data-testid="slash-command-output-messages"
      className="flex w-full flex-col"
    >
      {entries.map((entry) => {
        // @spec SC-002 — Inline help
        if (entry.kind === "help") {
          return (
            <GenericEventMessage
              key={entry.id}
              title={t(I18nKey.SLASH_COMMAND$HELP_TITLE, {
                count: entry.commands.length,
              })}
              details={
                <div className="mt-1 flex flex-col gap-2 text-sm text-neutral-300">
                  <div className="flex flex-col gap-1">
                    {entry.commands.map((item) => {
                      const description = item.descriptionKey
                        ? t(item.descriptionKey)
                        : (("description" in item.skill
                            ? item.skill.description
                            : null) ??
                          getSkillDescription(item.skill.content ?? ""));
                      return (
                        <div
                          key={item.command}
                          className="grid grid-cols-[minmax(7rem,auto)_1fr] gap-3 rounded-md border border-neutral-700 px-2 py-1.5"
                        >
                          <code className="text-neutral-100">
                            {item.command}
                          </code>
                          {description && <span>{description}</span>}
                        </div>
                      );
                    })}
                  </div>
                  <span>{t(I18nKey.SLASH_COMMAND$HELP_HINT)}</span>
                </div>
              }
              initiallyExpanded
            />
          );
        }

        // @spec SC-003 — Loaded extensions
        const hasExtensions =
          entry.skills.length > 0 ||
          entry.hooks.length > 0 ||
          entry.mcpServers.length > 0;
        return (
          <GenericEventMessage
            key={entry.id}
            title={
              hasExtensions
                ? SKILLS_COMMAND
                : t(I18nKey.SLASH_COMMAND$NO_SKILLS)
            }
            details={
              <div className="mt-1 flex flex-col gap-3 text-sm text-neutral-300">
                {entry.skills.length > 0 && (
                  <section className="flex flex-col gap-1">
                    <h4 className="font-medium text-neutral-100">
                      {t(I18nKey.SLASH_COMMAND$SKILLS_TITLE, {
                        count: entry.skills.length,
                      })}
                    </h4>
                    {entry.skills.map((skill) => {
                      const description = getSkillCardDescription(skill);
                      const pills = buildSkillPills(skill, t, {
                        testIdPrefix: "slash-skill-pill",
                      });
                      return (
                        <div
                          key={`${skill.source ?? "unknown"}-${skill.name}`}
                          data-testid={`slash-skill-card-${skill.name}`}
                          className={`flex min-w-0 items-start gap-3 p-3 ${extensionModuleCardSurfaceClassName}`}
                        >
                          <SkillIconBadge
                            skillName={skill.name}
                            className="h-8 w-8 rounded-md [&>svg]:h-4 [&>svg]:w-4"
                          />
                          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                            <div className="truncate font-medium text-neutral-100">
                              {skill.name}
                            </div>
                            {description && (
                              <p
                                data-testid={`slash-skill-description-${skill.name}`}
                                className="line-clamp-2 break-words text-xs leading-relaxed text-tertiary-light"
                              >
                                {description}
                              </p>
                            )}
                            <SkillCardPillRow
                              pills={pills}
                              testId={`slash-skill-pills-${skill.name}`}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </section>
                )}
                {entry.hooks.length > 0 && (
                  <section className="flex flex-col gap-1">
                    <h4 className="font-medium text-neutral-100">
                      {t(I18nKey.HOOKS_MODAL$TITLE)}
                    </h4>
                    {entry.hooks.map((hook) => {
                      const hookCount = hook.matchers.reduce(
                        (count, matcher) =>
                          count + (matcher.hooks?.length ?? 0),
                        0,
                      );
                      return (
                        <div
                          key={hook.event_type}
                          className="flex items-center justify-between rounded-md border border-neutral-700 px-2 py-1.5"
                        >
                          <span className="font-medium text-neutral-100">
                            {hook.event_type}
                          </span>
                          <span>
                            {t(I18nKey.HOOKS_MODAL$HOOK_COUNT, {
                              count: hookCount,
                            })}
                          </span>
                        </div>
                      );
                    })}
                  </section>
                )}
                {entry.mcpServers.length > 0 && (
                  <section className="flex flex-col gap-1">
                    <h4 className="font-medium text-neutral-100">
                      {t(I18nKey.MCP$INSTALLED_TITLE)}
                    </h4>
                    {entry.mcpServers.map((server) => (
                      <div
                        key={server.id}
                        className="grid grid-cols-[minmax(7rem,auto)_1fr] gap-3 rounded-md border border-neutral-700 px-2 py-1.5"
                      >
                        <span className="font-medium text-neutral-100">
                          {getInstalledServerTitle(server)}
                        </span>
                        <code>{server.type.toUpperCase()}</code>
                      </div>
                    ))}
                  </section>
                )}
                <span>{t(I18nKey.SLASH_COMMAND$SKILLS_HINT)}</span>
              </div>
            }
            initiallyExpanded
          />
        );
      })}
    </div>
  );
}
