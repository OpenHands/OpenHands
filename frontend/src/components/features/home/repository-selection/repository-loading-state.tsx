import { useTranslation } from "react-i18next";
import { Spinner } from "@heroui/react";
import { cn } from "#/utils/utils";

export interface RepositoryLoadingStateProps {
  wrapperClassName?: string;
}

export function RepositoryLoadingState({
  wrapperClassName,
}: RepositoryLoadingStateProps) {
  const { t } = useTranslation();
  return (
    <div
      data-testid="repo-dropdown-loading"
      className={cn(
        "flex items-center gap-2 max-w-[500px] h-10 px-3 bg-tertiary border border-stroke rounded-sm",
        wrapperClassName,
      )}
    >
      <Spinner size="sm" />
      <span className="text-sm">{t("HOME$LOADING_REPOSITORIES")}</span>
    </div>
  );
}
