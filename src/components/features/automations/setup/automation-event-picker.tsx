import { useEffect, useRef, useState } from "react";
import { useCombobox } from "downshift";
import { Check, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { parseAutomationEventKeys } from "#/api/automation-setup-events";
import { I18nKey } from "#/i18n/declaration";
import type { AutomationEventOption } from "#/manifests/types";
import { ComboboxCaretButton } from "#/ui/combobox-caret";
import {
  dropdownMenuRowClassName,
  dropdownTriggerShellClassName,
} from "#/utils/dropdown-classes";
import { cn } from "#/utils/utils";

const EVENT_KEY_INPUT_ID = "automation-setup-event-key";
const SELECTION_HINT_ID = "automation-event-selection-hint";

interface Props {
  value: string;
  options: AutomationEventOption[];
  onChange: (value: string) => void;
}

export function AutomationEventPicker({ value, options, onChange }: Props) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const controlRef = useRef<HTMLDivElement>(null);
  const [menuHeight, setMenuHeight] = useState(288);
  const selected = parseAutomationEventKeys(value);
  const allEvents: AutomationEventOption = {
    key: "*",
    label: t(I18nKey.AUTOMATION_SETUP$ALL_EVENTS),
    group: "",
    kind: "pattern",
  };
  const known = [allEvents, ...options.filter((option) => option.key !== "*")];
  const search = query.trim().toLowerCase();
  const items = known.filter((option) =>
    `${option.label} ${option.key} ${option.group}`
      .toLowerCase()
      .includes(search),
  );
  const customKey = query.trim();
  if (
    customKey &&
    !customKey.includes(",") &&
    !known.some((option) => option.key === customKey)
  ) {
    items.push({
      key: customKey,
      label: t(I18nKey.AUTOMATION_SETUP$USE_CUSTOM_EVENT, { key: customKey }),
      group: "",
      kind: "event",
    });
  }
  const toggle = (key: string) => {
    const next = selected.includes(key)
      ? selected.filter((item) => item !== key)
      : key === "*"
        ? [key]
        : [...selected.filter((item) => item !== "*"), key];
    onChange(next.join(", "));
  };
  const {
    isOpen,
    highlightedIndex,
    getInputProps,
    getMenuProps,
    getItemProps,
    getToggleButtonProps,
  } = useCombobox({
    items,
    inputValue: query,
    selectedItem: null,
    itemToString: (item) => item?.label ?? "",
    onInputValueChange: ({ inputValue }) => setQuery(inputValue),
    onSelectedItemChange: ({ selectedItem }) => {
      if (selectedItem) toggle(selectedItem.key);
    },
    stateReducer: (_state, { type, changes }) => {
      switch (type) {
        case useCombobox.stateChangeTypes.InputKeyDownEnter:
        case useCombobox.stateChangeTypes.ItemClick:
          return {
            ...changes,
            isOpen: true,
            inputValue: "",
            highlightedIndex: -1,
          };
        default:
          return changes;
      }
    },
  });
  useEffect(() => {
    if (!isOpen) return undefined;
    const resizeMenu = () => {
      const bottom = controlRef.current?.getBoundingClientRect().bottom ?? 0;
      setMenuHeight(
        Math.min(288, Math.max(120, window.innerHeight - bottom - 16)),
      );
    };
    resizeMenu();
    window.addEventListener("resize", resizeMenu);
    window.addEventListener("scroll", resizeMenu, true);
    return () => {
      window.removeEventListener("resize", resizeMenu);
      window.removeEventListener("scroll", resizeMenu, true);
    };
  }, [isOpen]);
  const labelFor = (key: string) =>
    known.find((option) => option.key === key)?.label ?? key;

  return (
    <div
      className="relative min-w-0 flex-1"
      data-testid="automation-event-picker"
    >
      {selected.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {selected.map((key) => (
            <span
              key={key}
              className="inline-flex min-w-0 max-w-full items-center gap-1 rounded-md border border-border bg-tertiary px-2 py-1 text-xs text-content"
              title={key}
            >
              <span className="truncate">{labelFor(key)}</span>
              <button
                type="button"
                aria-label={`${t(I18nKey.COMMON$REMOVE)} ${labelFor(key)}`}
                onClick={() => toggle(key)}
                className="shrink-0 rounded text-muted hover:text-contrast focus-visible:outline focus-visible:outline-2"
              >
                <X className="size-3.5" aria-hidden />
              </button>
            </span>
          ))}
        </div>
      )}
      <div ref={controlRef} className={dropdownTriggerShellClassName}>
        <input
          {...getInputProps({
            id: EVENT_KEY_INPUT_ID,
            "aria-label": t(I18nKey.AUTOMATION_SETUP$EVENT_KEY),
            "aria-describedby": SELECTION_HINT_ID,
            "data-testid": EVENT_KEY_INPUT_ID,
            placeholder: t(I18nKey.AUTOMATION_SETUP$SEARCH_EVENTS),
            className:
              "min-w-0 flex-1 bg-transparent text-sm text-contrast outline-none",
          })}
        />
        <ComboboxCaretButton
          {...getToggleButtonProps({
            "aria-label": t(I18nKey.AUTOMATION_SETUP$EVENT_KEY),
          })}
          isOpen={isOpen}
        />
      </div>
      <ul
        {...getMenuProps({
          "aria-multiselectable": true,
          style: { maxHeight: menuHeight },
          className: cn(
            "absolute z-50 mt-1 max-h-72 w-full overflow-auto rounded-md bg-tertiary p-1 context-menu-box-shadow",
            !isOpen && "hidden",
          ),
        })}
      >
        {isOpen &&
          items.map((option, index) => (
            <li
              key={option.key}
              {...getItemProps({ item: option, index })}
              role="option"
              aria-selected={selected.includes(option.key)}
              className={cn(
                dropdownMenuRowClassName,
                highlightedIndex === index && "bg-interactive-hover",
              )}
            >
              <Check
                className={cn(
                  "size-4 shrink-0",
                  !selected.includes(option.key) && "invisible",
                )}
                aria-hidden
              />
              <span className="min-w-0">
                <span className="block text-sm">{option.label}</span>
                <span className="block break-all font-mono text-xs text-muted">
                  {option.key}
                </span>
              </span>
            </li>
          ))}
      </ul>
      <p id={SELECTION_HINT_ID} className="mt-1.5 text-xs text-muted">
        {t(
          selected.includes("*")
            ? I18nKey.AUTOMATION_SETUP$ALL_EVENTS_HINT
            : I18nKey.AUTOMATION_SETUP$MULTIPLE_EVENTS_HINT,
        )}
      </p>
    </div>
  );
}
