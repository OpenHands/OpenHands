import { Loader } from "#/components/shared/loader";

export function RouteHydrateFallback() {
  return (
    <div className="flex min-h-[40vh] w-full items-center justify-center">
      <Loader size="large" className="text-tertiary-alt" />
    </div>
  );
}
