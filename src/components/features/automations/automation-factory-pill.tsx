import { Factory } from "lucide-react";
import {
  useFactories,
  useFactoriesSupported,
} from "#/hooks/query/use-factories";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

interface AutomationFactoryPillProps {
  factoryId: string;
  className?: string;
}

/** The name of the factory an automation is in. Nothing until the name is known. */
export function AutomationFactoryPill({
  factoryId,
  className,
}: AutomationFactoryPillProps) {
  const { isSupported } = useFactoriesSupported();
  const name = useFactories({ enabled: isSupported }).data?.find(
    (factory) => factory.id === factoryId,
  )?.name;
  if (!name) return null;

  return (
    <span
      data-testid="automation-factory-pill"
      title={name}
      className={cn(
        extensionModuleCardPillClassName,
        "max-w-40 gap-1",
        className,
      )}
    >
      <Factory className="size-3 shrink-0" aria-hidden />
      <span className="truncate">{name}</span>
    </span>
  );
}
