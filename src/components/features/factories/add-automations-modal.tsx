import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ErrorState } from "#/components/features/automations/error-state";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import {
  useAddAutomationsToFactory,
  useAutomations,
} from "#/hooks/query/use-automations";
import type { Factory } from "#/types/automation";
import { isDraftAutomation } from "#/utils/automation-state";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { modalTitleLgMediumClassName } from "#/utils/modal-classes";
import { cn } from "#/utils/utils";

interface AddAutomationsModalProps {
  factory: Factory;
  isOpen: boolean;
  onClose: () => void;
}

function AddAutomationsDialog({
  factory,
  onClose,
}: Omit<AddAutomationsModalProps, "isOpen">) {
  const { t } = useTranslation("openhands");
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [failed, setFailed] = useState<ReadonlySet<string>>(new Set());
  const {
    data,
    isLoading,
    isError,
    refetch,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useAutomations();
  const addAutomations = useAddAutomationsToFactory();

  // The list is filtered by name here, so every page is loaded up front; a
  // match would otherwise be missing until its page was fetched.
  useEffect(() => {
    if (hasNextPage && !isFetchingNextPage && !isError) void fetchNextPage();
  }, [hasNextPage, isFetchingNextPage, isError, fetchNextPage]);

  const candidates = useMemo(
    () =>
      (data?.automations ?? []).filter(
        (automation) =>
          !isDraftAutomation(automation) && !automation.factory_id,
      ),
    [data?.automations],
  );
  const needle = filter.trim().toLowerCase();
  const visible = candidates.filter((automation) =>
    automation.name.toLowerCase().includes(needle),
  );
  const selectedIds = candidates
    .filter((automation) => selected.has(automation.id))
    .map((automation) => automation.id);
  const allVisibleSelected =
    visible.length > 0 &&
    visible.every((automation) => selected.has(automation.id));
  const isPending = addAutomations.isPending;
  const isListLoading = isLoading || (hasNextPage && !isError);
  const title = t(I18nKey.FACTORIES$ADD_AUTOMATIONS);

  const toggle = (id: string, checked: boolean) =>
    setSelected((current) => {
      const next = new Set(current);
      if (checked) next.add(id);
      else next.delete(id);
      return next;
    });

  const toggleAllVisible = (checked: boolean) =>
    setSelected((current) => {
      const next = new Set(current);
      visible.forEach((automation) => {
        if (checked) next.add(automation.id);
        else next.delete(automation.id);
      });
      return next;
    });

  const handleAdd = () =>
    addAutomations.mutate(
      { ids: selectedIds, factoryId: factory.id },
      {
        onSuccess: ({ failedIds }) => {
          const added = selectedIds.length - failedIds.length;
          if (added > 0) {
            displaySuccessToast(
              t(I18nKey.FACTORIES$ADD_AUTOMATIONS_SUCCESS, {
                count: added,
                factory: factory.name,
              }),
            );
          }
          if (failedIds.length === 0) {
            onClose();
            return;
          }
          setSelected(new Set(failedIds));
          setFailed(new Set(failedIds));
          displayErrorToast(
            t(I18nKey.FACTORIES$ADD_AUTOMATIONS_PARTIAL, {
              failed: failedIds.length,
              total: selectedIds.length,
            }),
          );
        },
      },
    );

  const renderList = () => {
    if (isError && !data) {
      return <ErrorState onRetry={() => void refetch()} />;
    }
    if (isListLoading) {
      return (
        <div data-testid="add-automations-loading" className="py-6 text-center">
          <LoadingSpinner size="small" />
        </div>
      );
    }
    if (candidates.length === 0) {
      return (
        <p data-testid="add-automations-none" className="text-sm text-muted">
          {t(I18nKey.FACTORIES$ADD_AUTOMATIONS_NONE)}
        </p>
      );
    }
    if (visible.length === 0) {
      return (
        <p
          data-testid="add-automations-no-match"
          className="text-sm text-muted"
        >
          {t(I18nKey.FACTORIES$ADD_AUTOMATIONS_NO_MATCH)}
        </p>
      );
    }
    return (
      <>
        <label className="flex items-center gap-2 px-2 text-sm text-contrast">
          <input
            data-testid="add-automations-select-all"
            type="checkbox"
            checked={allVisibleSelected}
            disabled={isPending}
            onChange={(event) => toggleAllVisible(event.target.checked)}
          />
          {t(I18nKey.FACTORIES$ADD_AUTOMATIONS_SELECT_ALL)}
        </label>
        <ul className="flex max-h-72 flex-col overflow-y-auto">
          {visible.map((automation) => (
            <li key={automation.id}>
              <label className="flex items-center gap-3 rounded-lg px-2 py-1.5 hover:bg-surface-raised">
                <input
                  data-testid={`add-automations-check-${automation.id}`}
                  type="checkbox"
                  checked={selected.has(automation.id)}
                  disabled={isPending}
                  onChange={(event) =>
                    toggle(automation.id, event.target.checked)
                  }
                />
                <span
                  className="min-w-0 flex-1 truncate text-sm text-contrast"
                  title={automation.name}
                >
                  {automation.name}
                </span>
                {failed.has(automation.id) && (
                  <span className="shrink-0 text-xs text-feedback-error">
                    {t(I18nKey.FACTORIES$ADD_AUTOMATIONS_ROW_FAILED)}
                  </span>
                )}
              </label>
            </li>
          ))}
        </ul>
      </>
    );
  };

  return (
    <ModalBackdrop onClose={onClose} aria-label={title}>
      <div
        data-testid="add-automations-modal"
        className={cn(
          "relative flex w-full flex-col gap-4 rounded-xl border border-border bg-surface p-6",
          modalWidthClassName("md"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-4 text-muted hover:text-foreground"
          aria-label={t(I18nKey.BUTTON$CLOSE)}
        >
          <XMarkIcon className="size-5" />
        </button>
        <div>
          <h2 className={modalTitleLgMediumClassName}>{title}</h2>
          <p className="mt-2 text-sm text-muted">
            {t(I18nKey.FACTORIES$ADD_AUTOMATIONS_MESSAGE, {
              name: factory.name,
            })}
          </p>
        </div>
        <input
          data-testid="add-automations-filter"
          type="text"
          value={filter}
          placeholder={t(I18nKey.FACTORIES$ADD_AUTOMATIONS_FILTER)}
          aria-label={t(I18nKey.FACTORIES$ADD_AUTOMATIONS_FILTER)}
          onChange={(event) => setFilter(event.target.value)}
          className="rounded-lg border border-border bg-base-secondary px-3 py-2 text-sm text-content"
        />
        {renderList()}
        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-border px-4 py-2 text-sm text-contrast hover:bg-surface-raised"
          >
            {t(I18nKey.AUTOMATIONS$CANCEL)}
          </button>
          <BrandButton
            type="button"
            variant="primary"
            testId="add-automations-submit"
            isDisabled={selectedIds.length === 0 || isPending}
            onClick={handleAdd}
          >
            {t(I18nKey.FACTORIES$ADD_AUTOMATIONS_SUBMIT, {
              count: selectedIds.length,
            })}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}

/**
 * Put several of the org's automations in one factory at once: filter them by
 * name, tick the ones to add, and confirm. Closed, it renders nothing and
 * holds no selection.
 */
export function AddAutomationsModal({
  factory,
  isOpen,
  onClose,
}: AddAutomationsModalProps) {
  if (!isOpen) return null;
  return <AddAutomationsDialog factory={factory} onClose={onClose} />;
}
