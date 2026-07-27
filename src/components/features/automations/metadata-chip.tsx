import { cn } from "#/utils/utils";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";

interface MetadataChipProps {
  icon: React.ReactNode;
  label: string;
}

export function MetadataChip({ icon, label }: MetadataChipProps) {
  return (
    <span className={cn(extensionModuleCardPillClassName, "gap-1")}>
      {icon}
      {label}
    </span>
  );
}
