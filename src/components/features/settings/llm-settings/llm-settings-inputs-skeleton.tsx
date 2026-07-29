import { InputSkeleton } from "../input-skeleton";
import { SubtextSkeleton } from "../subtext-skeleton";
import { SwitchSkeleton } from "../switch-skeleton";

export function LlmSettingsInputsSkeleton() {
  return (
    <div
      data-testid="app-settings-skeleton"
      className="px-11 py-9 flex flex-col gap-6"
    >
      <SwitchSkeleton />
      <InputSkeleton />
      <InputSkeleton />
      <InputSkeleton />
      <SubtextSkeleton />
      <SwitchSkeleton />
      <SwitchSkeleton />
      <InputSkeleton />
    </div>
  );
}
