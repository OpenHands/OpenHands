import { Autocomplete, AutocompleteItem } from "@heroui/react";
import React, { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { OptionalTag } from "./optional-tag";
import { cn } from "#/utils/utils";

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
  isClearable,
  allowsCustomValue,
  required,
  onSelectionChange,
  onInputChange,
  defaultFilter,
  startContent,
  inputWrapperClassName,
  inputClassName,
}: SettingsDropdownInputProps) {
  const { t } = useTranslation();

  return (
    <label className={cn("flex flex-col gap-2.5", wrapperClassName)}>
      {label && (
        <div className="flex items-center gap-1">
          <span className="text-sm">{label}</span>
          {showOptionalTag && <OptionalTag />}
        </div>
      )}
      <Autocomplete
        aria-label={typeof label === "string" ? label : name}
        data-testid={testId}
        name={name}
        defaultItems={items}
        defaultSelectedKey={defaultSelectedKey}
        selectedKey={selectedKey}
        onSelectionChange={onSelectionChange}
        onInputChange={onInputChange}
        isClearable={isClearable}
        isDisabled={isDisabled || isLoading}
        isLoading={isLoading}
        placeholder={isLoading ? t("HOME$LOADING") : placeholder}
        allowsCustomValue={allowsCustomValue}
        isRequired={required}
        className="w-full"
        classNames={{
          popoverContent: "bg-tertiary rounded-xl",
        }}
        inputProps={{
          classNames: {
            inputWrapper: cn(
              "bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic",
              inputWrapperClassName,
            ),
            input: inputClassName,
          },
        }}
        defaultFilter={defaultFilter}
        startContent={startContent || null}
      >
        {(item) => (
          <AutocompleteItem key={item.key}>{item.label}</AutocompleteItem>
        )}
      </Autocomplete>
    </label>
  );
}
