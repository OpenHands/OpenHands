import { ComboBox, Input, ListBox } from "@heroui/react";
import { ComboBoxStateContext } from "react-aria-components";
import React, { ReactNode, useCallback, useContext } from "react";
import { useTranslation } from "react-i18next";
import { OptionalTag } from "./optional-tag";
import { cn } from "#/utils/utils";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { heroUiAutocompleteSelectorButtonClassName } from "#/ui/combobox-caret";
import { I18nKey } from "#/i18n/declaration";

interface SettingsDropdownInputProps {
  testId: string;
  name: string;
  items: { key: React.Key; label: string }[];
  label?: ReactNode;
  wrapperClassName?: string;
  placeholder?: string;
  showOptionalTag?: boolean;
  isDisabled?: boolean;
  isLoading?: boolean;
  defaultSelectedKey?: string;
  selectedKey?: string;
  isClearable?: boolean;
  allowsCustomValue?: boolean;
  required?: boolean;
  onSelectionChange?: (key: React.Key | null) => void;
  onInputChange?: (value: string) => void;
  defaultFilter?: (textValue: string, inputValue: string) => boolean;
  startContent?: ReactNode;
  inputWrapperClassName?: string;
  inputClassName?: string;
}

function ClearButton({
  isClearable,
  isLoading,
  clearLabel,
}: {
  isClearable: boolean;
  isLoading: boolean | undefined;
  clearLabel: string;
}) {
  const state = useContext(ComboBoxStateContext);
  const handleClear = useCallback(() => {
    state?.setSelectedKey(null);
    state?.setInputValue("");
  }, [state]);

  if (!isClearable || isLoading) return null;

  return (
    <button
      type="button"
      aria-label={clearLabel}
      onClick={handleClear}
      className="inline-flex size-4 shrink-0 items-center justify-center text-current opacity-60 transition-opacity hover:opacity-100"
    >
      ×
    </button>
  );
}

export function SettingsDropdownInput({
  testId,
  label,
  wrapperClassName,
  name,
  items,
  placeholder,
  showOptionalTag,
  isDisabled,
  isLoading,
  defaultSelectedKey,
  selectedKey,
  isClearable = false,
  allowsCustomValue,
  required,
  onSelectionChange,
  onInputChange,
  defaultFilter,
  startContent,
  inputWrapperClassName,
  inputClassName,
}: SettingsDropdownInputProps) {
  const { t } = useTranslation("openhands");

  return (
    <label
      className={cn("flex flex-col gap-2.5 w-full min-w-0", wrapperClassName)}
    >
      {label && (
        <div className="flex items-center gap-1">
          <span className="text-sm">{label}</span>
          {showOptionalTag && <OptionalTag />}
        </div>
      )}
      <ComboBox
        aria-label={typeof label === "string" ? label : name}
        defaultItems={items}
        defaultSelectedKey={defaultSelectedKey}
        selectedKey={selectedKey}
        onSelectionChange={onSelectionChange}
        onInputChange={onInputChange}
        isDisabled={isDisabled || isLoading}
        allowsCustomValue={allowsCustomValue}
        isRequired={required}
        className="w-full"
        defaultFilter={defaultFilter}
      >
        <ComboBox.InputGroup className={cn(inputWrapperClassName)}>
          {startContent || null}
          <Input
            data-testid={testId}
            name={name}
            className={cn(formControlSettingsFieldClassName, inputClassName)}
            placeholder={isLoading ? t(I18nKey.HOME$LOADING) : placeholder}
          />
          <ClearButton
            isClearable={isClearable}
            isLoading={isLoading}
            clearLabel={t(I18nKey.COMMON$CLEAR_SELECTION)}
          />
          <ComboBox.Trigger
            className={heroUiAutocompleteSelectorButtonClassName}
          />
        </ComboBox.InputGroup>
        <ComboBox.Popover className="rounded-xl">
          <ListBox items={items}>
            {(item) => (
              <ListBox.Item
                id={item.key as string | number}
                textValue={item.label}
              >
                {item.label}
              </ListBox.Item>
            )}
          </ListBox>
        </ComboBox.Popover>
      </ComboBox>
    </label>
  );
}
