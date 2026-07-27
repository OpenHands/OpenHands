import { InputSkeleton } from "../input-skeleton";
import { SwitchSkeleton } from "../switch-skeleton";

export function AppSettingsInputsSkeleton() {
  return (
    <div
      data-testid="app-settings-skeleton"
      className="px-11 py-9 flex flex-col gap-6"
    >
      <InputSkeleton />
      <SwitchSkeleton />
      <SwitchSkeleton />
    </div>
  );
}
