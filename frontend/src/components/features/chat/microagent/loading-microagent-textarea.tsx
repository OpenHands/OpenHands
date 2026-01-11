import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";

export function LoadingMicroagentTextarea() {
  const { t } = useTranslation();

  return (
    <textarea
      required
      disabled
      defaultValue=""
      placeholder={t("MICROAGENT$LOADING_PROMPT")}
      rows={6}
      className={cn(
        "bg-tertiary border border-stroke w-full rounded p-2 placeholder:italic placeholder:text-tertiary-alt resize-none",
        "disabled:bg-surface-disabled disabled:border-surface-disabled disabled:cursor-not-allowed",
      )}
    />
  );
}
