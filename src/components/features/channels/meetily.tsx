import React from "react";
import { useTranslation } from "react-i18next";
import { KANBAN_PATH } from "#/api/kanban-service/kanban-constants";
import type {
  MeetilyIngestResult,
  MeetilyPreview,
} from "#/api/meetily-service/meetily-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useKanbanBoards } from "#/hooks/query/use-kanban";
import {
  useIngestTranscript,
  usePreviewTranscript,
} from "#/hooks/query/use-meetily";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

export function MeetilyImport() {
  const { t } = useTranslation("openhands");
  const boardsQuery = useKanbanBoards();
  const previewMutation = usePreviewTranscript();
  const ingestMutation = useIngestTranscript();
  const [text, setText] = React.useState("");
  const [preview, setPreview] = React.useState<MeetilyPreview | null>(null);
  const [result, setResult] = React.useState<MeetilyIngestResult | null>(null);
  const boards = boardsQuery.data ?? [];
  const boardId = boards[0]?.id;

  const payload = { text, board_id: boardId };

  return (
    <section data-testid="meetily-import" className="flex flex-col gap-3">
      <Typography variant="h2">{t(I18nKey.MEETILY$TITLE)}</Typography>
      <label className="text-sm text-tertiary-light" htmlFor="meetily-paste">
        {t(I18nKey.MEETILY$PASTE)}
      </label>
      <textarea
        id="meetily-paste"
        data-testid="meetily-paste"
        className="min-h-32 rounded-xl bg-base-secondary p-3 text-sm text-white"
        value={text}
        onChange={(event) => setText(event.target.value)}
      />
      <label className="text-sm text-tertiary-light" htmlFor="meetily-file">
        {t(I18nKey.MEETILY$FILE)}
      </label>
      <input
        id="meetily-file"
        data-testid="meetily-file"
        type="file"
        accept=".json,.md,.csv,.txt,text/plain,text/markdown,text/csv,application/json"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          setText(await file.text());
        }}
      />
      <div className="flex flex-wrap gap-2">
        <BrandButton
          type="button"
          variant="tertiary"
          testId="meetily-preview"
          isDisabled={!text.trim()}
          onClick={async () => {
            const data = await previewMutation.mutateAsync(payload);
            setPreview(data);
            setResult(null);
          }}
        >
          {t(I18nKey.MEETILY$PREVIEW)}
        </BrandButton>
        <BrandButton
          type="button"
          variant="primary"
          testId="meetily-create"
          isDisabled={!text.trim() || !boardId}
          onClick={async () => {
            const data = await ingestMutation.mutateAsync(payload);
            setResult(data);
            setPreview(data);
          }}
        >
          {t(I18nKey.MEETILY$CREATE)}
        </BrandButton>
      </div>
      {!boardId ? (
        <p className="text-sm text-tertiary-light">
          {t(I18nKey.MEETILY$NO_BOARD)}
        </p>
      ) : (
        <p className="text-sm text-tertiary-light">
          {t(I18nKey.MEETILY$BOARD)} {boards[0]?.name}
        </p>
      )}
      {preview?.summary ? (
        <p data-testid="meetily-summary" className="text-sm text-white">
          {t(I18nKey.MEETILY$SUMMARY)} {preview.summary}
        </p>
      ) : null}
      {(preview?.items.length ?? 0) === 0 && preview ? (
        <p className="text-sm text-tertiary-light">
          {t(I18nKey.MEETILY$EMPTY)}
        </p>
      ) : null}
      {(preview?.items.length ?? 0) > 0 ? (
        <ul data-testid="meetily-items" className="flex flex-col gap-2">
          {preview?.items.map((item) => (
            <li
              key={`${item.card_type}-${item.title}`}
              data-testid="meetily-item"
              className="rounded-xl bg-base-secondary p-3 text-sm text-white"
            >
              <span className={cn(extensionModuleCardPillClassName, "mr-2")}>
                {t(I18nKey.MEETILY$CARD_TYPE)} {item.card_type}
              </span>
              {item.title}
            </li>
          ))}
        </ul>
      ) : null}
      {(preview?.duplicates?.length ?? 0) > 0 ? (
        <ul data-testid="meetily-duplicates" className="flex flex-col gap-2">
          {preview?.duplicates?.map((dup) => (
            <li
              key={dup.existing_card_id}
              data-testid="meetily-duplicate"
              className="rounded-xl bg-base-secondary p-3 text-sm text-tertiary-light"
            >
              {t(I18nKey.MEETILY$DUPLICATE)} {dup.existing_title}
            </li>
          ))}
        </ul>
      ) : null}
      {result ? (
        <p data-testid="meetily-created" className="text-sm text-white">
          {t(I18nKey.MEETILY$CREATED)} {result.created.length}
        </p>
      ) : null}
      {result ? (
        <NavigationLink
          to={KANBAN_PATH}
          data-testid="meetily-open-board"
          className="text-sm text-primary"
        >
          {t(I18nKey.MEETILY$OPEN_BOARD)}
        </NavigationLink>
      ) : null}
    </section>
  );
}
