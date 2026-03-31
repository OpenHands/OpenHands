import { cn } from "#/utils/utils";

type SettingsNavDividerProps = {
  className?: string;
};

export function SettingsNavDivider({ className }: SettingsNavDividerProps) {
  return <div className={cn("border-t border-[#242424] w-full", className)} />;
}
