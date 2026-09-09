import React from "react";
import { Pencil, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { NavigationLink } from "#/components/shared/navigation-link";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import {
  settingsListIconActionButtonClassName,
  settingsListScrollContainerClassName,
  settingsListTableCellClassName,
  settingsListTableHeadClassName,
  settingsListTableHeaderCellClassName,
  settingsListTableRowClassName,
} from "#/utils/settings-list-classes";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { unusedCatalogEntries, type Macro } from "#/utils/macros";
import { useMacrosStore } from "#/stores/macros-store";

type MacrosView = "list" | "add" | "edit";

export function MacrosManager() {
  const { t } = useTranslation("openhands");
  const macros = useMacrosStore((state) => state.macros);
  const addMacro = useMacrosStore((state) => state.addMacro);
  const updateMacro = useMacrosStore((state) => state.updateMacro);
  const removeMacro = useMacrosStore((state) => state.removeMacro);
  const addFromCatalog = useMacrosStore((state) => state.addFromCatalog);

  const [view, setView] = React.useState<MacrosView>("list");
  const [editing, setEditing] = React.useState<Macro | null>(null);
  const [title, setTitle] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [macroToDelete, setMacroToDelete] = React.useState<Macro | null>(null);

  const catalog = unusedCatalogEntries(macros);
  const isFormView = view === "add" || view === "edit";
  const canSave = title.trim().length > 0 && prompt.trim().length > 0;

  const openAdd = () => {
    setEditing(null);
    setTitle("");
    setPrompt("");
    setView("add");
  };

  const openEdit = (macro: Macro) => {
    setEditing(macro);
    setTitle(t(macro.title as I18nKey));
    setPrompt(macro.prompt);
    setView("edit");
  };

  const backToList = () => {
    setView("list");
    setEditing(null);
    setTitle("");
    setPrompt("");
  };

  const handleSave = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSave) {
      return;
    }
    if (view === "edit" && editing) {
      const nextTitle =
        title.trim() === t(editing.title as I18nKey)
          ? editing.title
          : title.trim();
      updateMacro(editing.id, { title: nextTitle, prompt: prompt.trim() });
    } else {
      addMacro({ title: title.trim(), prompt: prompt.trim() });
    }
    backToList();
  };

  return (
    <div data-testid="macros-settings-screen" className="flex flex-col gap-6">
      {view === "list" ? (
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 space-y-1">
            <Typography.H2>{t(I18nKey.SETTINGS$NAV_MACROS)}</Typography.H2>
            <p
              data-testid="settings-page-subtitle"
              className="text-sm leading-5 text-tertiary-light"
            >
              {t(I18nKey.SETTINGS$PAGE_MACROS_SUBLINE)}
            </p>
          </div>
          <BrandButton
            testId="add-macro-button"
            type="button"
            variant="primary"
            className="shrink-0 whitespace-nowrap"
            onClick={openAdd}
          >
            {t(I18nKey.SETTINGS$MACROS_ADD)}
          </BrandButton>
        </div>
      ) : null}

      {isFormView ? (
        <div className="flex flex-col gap-2">
          <BackNavButton testId="back-to-macros" onClick={backToList}>
            {t(I18nKey.BUTTON$BACK)}
          </BackNavButton>
          <Typography.H2>
            {view === "add"
              ? t(I18nKey.SETTINGS$MACROS_ADD)
              : t(I18nKey.SETTINGS$MACROS_EDIT)}
          </Typography.H2>
        </div>
      ) : null}

      {view === "list" && macros.length === 0 ? (
        <div
          data-testid="macros-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.SETTINGS$MACROS_EMPTY)}
          </p>
        </div>
      ) : null}

      {view === "list" && macros.length > 0 ? (
        <div
          className={settingsListScrollContainerClassName}
          data-testid="macros-list"
        >
          <table className="w-full table-fixed">
            <thead className={settingsListTableHeadClassName}>
              <tr>
                <th className={settingsListTableHeaderCellClassName}>
                  {t(I18nKey.SETTINGS$NAME)}
                </th>
                <th className="w-20" />
              </tr>
            </thead>
            <tbody>
              {macros.map((macro) => (
                <tr
                  key={macro.id}
                  data-testid={`macro-item-${macro.id}`}
                  className={settingsListTableRowClassName}
                >
                  <td
                    className={cn(
                      settingsListTableCellClassName,
                      "text-content-2 truncate",
                    )}
                  >
                    {t(macro.title as I18nKey)}
                  </td>
                  <td className={settingsListTableCellClassName}>
                    <div className="flex items-center justify-end gap-0.5">
                      <button
                        data-testid={`edit-macro-${macro.id}`}
                        type="button"
                        onClick={() => openEdit(macro)}
                        aria-label={t(I18nKey.SETTINGS$MACROS_EDIT)}
                        className={settingsListIconActionButtonClassName}
                      >
                        <Pencil
                          aria-hidden
                          className="size-4"
                          strokeWidth={2}
                        />
                      </button>
                      <button
                        data-testid={`delete-macro-${macro.id}`}
                        type="button"
                        onClick={() => setMacroToDelete(macro)}
                        aria-label={t(I18nKey.BUTTON$DELETE)}
                        className={settingsListIconActionButtonClassName}
                      >
                        <Trash2
                          aria-hidden
                          className="size-4"
                          strokeWidth={2}
                        />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {view === "list" && catalog.length > 0 ? (
        <section className="flex flex-col gap-3" data-testid="macros-catalog">
          <Typography.H3>{t(I18nKey.SETTINGS$MACROS_CATALOG)}</Typography.H3>
          <div className={settingsListScrollContainerClassName}>
            <table className="w-full table-fixed">
              <tbody>
                {catalog.map((entry) => (
                  <tr key={entry.id} className={settingsListTableRowClassName}>
                    <td
                      className={cn(
                        settingsListTableCellClassName,
                        "text-content-2 truncate",
                      )}
                    >
                      {t(entry.titleKey as I18nKey)}
                    </td>
                    <td className={cn(settingsListTableCellClassName, "w-28")}>
                      <div className="flex justify-end">
                        <BrandButton
                          testId={`add-catalog-macro-${entry.id}`}
                          type="button"
                          variant="secondary"
                          onClick={() => addFromCatalog(entry.id)}
                        >
                          {t(I18nKey.BUTTON$ADD)}
                        </BrandButton>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {view === "list" ? (
        <p className="text-sm text-tertiary-light">
          <NavigationLink
            to="/automations"
            data-testid="macros-automations-link"
            className="underline underline-offset-2 hover:text-[var(--oh-foreground)]"
          >
            {t(I18nKey.SETTINGS$MACROS_AUTOMATIONS_HINT)}
          </NavigationLink>
        </p>
      ) : null}

      {isFormView ? (
        <form
          data-testid={view === "add" ? "add-macro-form" : "edit-macro-form"}
          onSubmit={handleSave}
          className="flex flex-col items-start gap-6"
        >
          <SettingsInput
            testId="macro-title-input"
            name="macro-title"
            type="text"
            label={t(I18nKey.SETTINGS$NAME)}
            className="w-full min-w-0"
            required
            value={title}
            onChange={setTitle}
          />
          <label className="flex flex-col gap-2.5 w-full min-w-0">
            <span className="text-sm">{t(I18nKey.SETTINGS$MACROS_PROMPT)}</span>
            <textarea
              data-testid="macro-prompt-input"
              name="macro-prompt"
              value={prompt}
              required
              onChange={(event) => setPrompt(event.currentTarget.value)}
              className={cn(
                "min-h-40 resize-y",
                formControlMultilineFieldClassName,
              )}
            />
          </label>
          <div className="flex items-center gap-4">
            <BrandButton
              testId="cancel-macro-button"
              type="button"
              variant="secondary"
              onClick={backToList}
            >
              {t(I18nKey.BUTTON$CANCEL)}
            </BrandButton>
            <BrandButton
              testId="save-macro-button"
              type="submit"
              variant="primary"
              isDisabled={!canSave}
            >
              {t(I18nKey.BUTTON$SAVE)}
            </BrandButton>
          </div>
        </form>
      ) : null}

      {macroToDelete ? (
        <ConfirmationModal
          text={t(I18nKey.SETTINGS$MACROS_CONFIRM_DELETE)}
          onConfirm={() => {
            removeMacro(macroToDelete.id);
            setMacroToDelete(null);
          }}
          onCancel={() => setMacroToDelete(null)}
        />
      ) : null}
    </div>
  );
}
