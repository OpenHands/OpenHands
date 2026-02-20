import React, { useMemo } from "react";
import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { useCombobox } from "downshift";
import { I18nKey } from "#/i18n/declaration";
import GlobeIcon from "#/icons/globe.svg?react";
import { cn } from "#/utils/utils";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { Card } from "#/ui/card";
import { BrandButton } from "../../settings/brand-button";
import { useConnectToEnvironment } from "#/hooks/mutation/use-connect-to-environment";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { ToggleButton } from "../shared/toggle-button";
import { ClearButton } from "../shared/clear-button";
import { GenericDropdownMenu } from "../shared/generic-dropdown-menu";
import { DropdownItem } from "../shared/dropdown-item";
import { EmptyState } from "../shared/empty-state";

const STORAGE_KEY = "environment-url-history";
const MAX_HISTORY = 10;

function getUrlHistory(): string[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function addUrlToHistory(url: string) {
  const history = getUrlHistory().filter((u) => u !== url);
  history.unshift(url);
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(history.slice(0, MAX_HISTORY)),
  );
}

function removeUrlFromHistory(url: string) {
  const history = getUrlHistory().filter((u) => u !== url);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

function isValidEnvironmentUrl(url: string): {
  valid: boolean;
  errorKey?: I18nKey;
} {
  if (!url.trim()) {
    return { valid: false, errorKey: I18nKey.HOME$ENVIRONMENT_URL_REQUIRED };
  }
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return {
        valid: false,
        errorKey: I18nKey.HOME$ENVIRONMENT_URL_INVALID_SCHEME,
      };
    }
    if (!parsed.hostname) {
      return { valid: false, errorKey: I18nKey.HOME$ENVIRONMENT_URL_INVALID };
    }
    return { valid: true };
  } catch {
    return { valid: false, errorKey: I18nKey.HOME$ENVIRONMENT_URL_INVALID };
  }
}

export function EnvironmentConnector() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [validationError, setValidationError] = React.useState<string | null>(
    null,
  );
  const [urlHistory, setUrlHistory] = React.useState(getUrlHistory);
  const [inputValue, setInputValue] = React.useState("");

  const {
    mutate: connectToEnvironment,
    isPending,
    isSuccess,
  } = useConnectToEnvironment();
  const isCreatingConversationElsewhere = useIsCreatingConversation();

  const isConnecting =
    isPending || isSuccess || isCreatingConversationElsewhere;

  const filteredHistory = useMemo(
    () =>
      urlHistory.filter(
        (u) =>
          !inputValue.trim() ||
          u.toLowerCase().includes(inputValue.toLowerCase()),
      ),
    [urlHistory, inputValue],
  );

  const {
    isOpen,
    getToggleButtonProps,
    getMenuProps,
    getInputProps,
    getItemProps,
    highlightedIndex,
    selectedItem,
  } = useCombobox({
    items: filteredHistory,
    inputValue,
    itemToString: (item) => item ?? "",
    onSelectedItemChange: ({ selectedItem: newItem }) => {
      if (newItem) {
        setInputValue(newItem);
        setValidationError(null);
      }
    },
    onInputValueChange: ({ inputValue: newValue, type }) => {
      if (type === useCombobox.stateChangeTypes.InputChange) {
        setInputValue(newValue ?? "");
      }
      setValidationError(null);
    },
  });

  const handleClear = () => {
    setInputValue("");
    setValidationError(null);
  };

  const handleConnect = () => {
    const validation = isValidEnvironmentUrl(inputValue);
    if (!validation.valid) {
      setValidationError(validation.errorKey ? t(validation.errorKey) : null);
      return;
    }
    setValidationError(null);

    const url = inputValue.trim();
    addUrlToHistory(url);
    setUrlHistory(getUrlHistory());

    connectToEnvironment(
      { environmentUrl: url },
      {
        onSuccess: (data) => {
          sessionStorage.setItem(
            `environment-url:${data.conversation_id}`,
            url,
          );
          navigate(`/conversations/${data.conversation_id}`);
        },
      },
    );
  };

  const handleRemoveFromHistory = (
    e: React.MouseEvent,
    urlToRemove: string,
  ) => {
    e.stopPropagation();
    e.preventDefault();
    removeUrlFromHistory(urlToRemove);
    setUrlHistory(getUrlHistory());
  };

  const renderItem = (
    item: string,
    index: number,
    currentHighlightedIndex: number,
    currentSelectedItem: string | null,
    currentGetItemProps: typeof getItemProps,
  ) => (
    <div key={item} className="flex items-center">
      <div className="flex-1 min-w-0">
        <DropdownItem
          item={item}
          index={index}
          isSelected={item === currentSelectedItem}
          getItemProps={currentGetItemProps}
          getDisplayText={(u) => u}
          getItemKey={(u) => u}
          renderIcon={() => (
            <GlobeIcon
              width={14}
              height={14}
              className="text-[#A3A3A3] shrink-0"
            />
          )}
        />
      </div>
      <button
        type="button"
        className="text-[#A3A3A3] hover:text-white shrink-0 p-1 mr-1"
        onClick={(e) => handleRemoveFromHistory(e, item)}
        onMouseDown={(e) => e.preventDefault()}
        aria-label="Remove from history"
      >
        <svg
          className="w-3.5 h-3.5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>
    </div>
  );

  const renderEmptyState = (currentInputValue: string) => (
    <EmptyState
      inputValue={currentInputValue}
      searchMessage={t(I18nKey.HOME$NO_MATCHING_ENVIRONMENTS)}
      emptyMessage={t(I18nKey.HOME$NO_MATCHING_ENVIRONMENTS)}
      testId="environment-url-dropdown-empty"
    />
  );

  return (
    <Card>
      <CardTitle icon={<GlobeIcon width={17} height={14} />}>
        {t(I18nKey.HOME$CONNECT_TO_ENVIRONMENT)}
      </CardTitle>
      <Typography.Text>
        {t(I18nKey.HOME$CONNECT_TO_ENVIRONMENT_DESCRIPTION)}
      </Typography.Text>

      <div className="flex flex-col gap-2">
        <label className="text-xs text-[#A3A3A3]">
          {t(I18nKey.HOME$ENVIRONMENT_URL_LABEL)}
        </label>
        <div className="relative">
          <div className="absolute left-2 top-1/2 -translate-y-1/2 z-10">
            <GlobeIcon width={16} height={16} className="text-[#A3A3A3]" />
          </div>
          <input
            // eslint-disable-next-line react/jsx-props-no-spreading
            {...getInputProps({
              placeholder: "https://your-environment.example.com",
              disabled: isConnecting,
              className: cn(
                "w-full px-3 py-2 border border-[#727987] rounded-sm shadow-none h-[42px] min-h-[42px] max-h-[42px]",
                "bg-[#454545] text-[#A3A3A3] placeholder:text-[#A3A3A3]",
                "focus:outline-none focus:ring-0 focus:border-[#727987]",
                "disabled:bg-[#363636] disabled:cursor-not-allowed disabled:opacity-60",
                "pl-7 pr-16 text-sm font-normal leading-5",
              ),
              onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
                setInputValue(e.target.value);
              },
            })}
            data-testid="environment-url-input"
          />
          <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center">
            {inputValue && (
              <ClearButton
                disabled={isConnecting}
                onClear={handleClear}
                testId="environment-url-clear"
              />
            )}
            <ToggleButton
              isOpen={isOpen}
              disabled={isConnecting}
              getToggleButtonProps={getToggleButtonProps}
              iconClassName="w-10 h-10"
            />
          </div>
        </div>

        <GenericDropdownMenu
          isOpen={isOpen && filteredHistory.length > 0}
          filteredItems={filteredHistory}
          inputValue={inputValue}
          highlightedIndex={highlightedIndex}
          selectedItem={selectedItem}
          getMenuProps={getMenuProps}
          getItemProps={getItemProps}
          renderItem={renderItem}
          renderEmptyState={renderEmptyState}
          itemKey={(url) => url}
          testId="environment-url-dropdown-menu"
        />

        {validationError && (
          <span className="text-xs text-red-400">{validationError}</span>
        )}
      </div>

      <BrandButton
        testId="connect-to-environment-button"
        variant="primary"
        type="button"
        onClick={handleConnect}
        isDisabled={isConnecting || !inputValue.trim()}
        className="w-auto absolute bottom-5 left-5 right-5 font-semibold"
      >
        {!isConnecting && t(I18nKey.HOME$CONNECT)}
        {isConnecting && t(I18nKey.HOME$CONNECTING)}
      </BrandButton>
    </Card>
  );
}
