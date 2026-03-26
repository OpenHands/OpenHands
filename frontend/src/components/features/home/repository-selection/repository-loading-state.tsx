import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

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
        "flex items-center gap-2 max-w-[500px] h-10 px-3 bg-tertiary border border-[#717888] rounded-sm",
        wrapperClassName,
      )}
    >
      <LoadingSpinner className="w-4 h-4" />
      <span className="text-sm">{t("HOME$LOADING_REPOSITORIES")}</span>
    </div>
  );
}
