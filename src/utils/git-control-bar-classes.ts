import { cn } from "#/utils/utils";

export const gitControlBarActionButtonBaseClassName =
  "flex flex-row gap-1 items-center justify-center rounded-[100px]";

export function gitControlBarActionButtonClassName(isEnabled: boolean) {
  return cn(
    gitControlBarActionButtonBaseClassName,
    isEnabled
      ? "bg-surface hover:bg-tertiary cursor-pointer text-white"
      : "bg-surface cursor-not-allowed opacity-50 text-muted",
  );
}

export function gitControlBarActionIconColor(isEnabled: boolean) {
  return isEnabled ? "white" : "var(--oh-muted)";
}

export const gitControlBarActionLabelClassName =
  "font-normal text-sm leading-5 truncate";
