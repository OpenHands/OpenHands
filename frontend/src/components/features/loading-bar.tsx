import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";

function LoadingBar() {
  return (
    <div className="flex flex-col items-center gap-4 w-full px-4">
      <OpenHandsLogoWhite className="w-24 sm:w-28 md:w-32 h-auto" />

      {/* Responsive loader bar */}
      <div className="w-full max-w-xs h-2 sm:h-2.5 bg-[#474A54] rounded border border-[#474A54] overflow-hidden">
        <div className="h-full w-1/3 bg-white rounded" />
      </div>
    </div>
  );
}

export default LoadingBar;
