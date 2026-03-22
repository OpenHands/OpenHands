import { useTranslation } from "react-i18next";
import { Spinner } from "#/components/shared/spinner";
import { cn } from "#/utils/utils";

interface BranchLoadingStateProps {
  wrapperClassName?: string;
}

export function BranchLoadingState({
  wrapperClassName,
}: BranchLoadingStateProps) {
  const { t } = useTranslation();
  return (
    <div
      data-testid="branch-dropdown-loading"
      className={cn(
        "flex items-center gap-2 max-w-[500px] h-10 px-3 bg-tertiary border border-[#717888] rounded-sm",
        wrapperClassName,
      )}
    >
      <Spinner size="sm" />
      <span className="text-sm">{t("HOME$LOADING_BRANCHES")}</span>
    </div>
  );
}
