import GlobeIcon from "#/icons/globe.svg?react";

interface EnvironmentLinkProps {
  url: string;
}

export function EnvironmentLink({ url }: EnvironmentLinkProps) {
  let displayUrl: string;
  try {
    const parsed = new URL(url);
    displayUrl = parsed.hostname;
  } catch {
    displayUrl = url;
  }

  return (
    <div className="flex items-center gap-1 text-xs text-[#A3A3A3]">
      <GlobeIcon width={14} height={14} className="text-[#A3A3A3] shrink-0" />
      <span className="truncate" title={url}>
        {displayUrl}
      </span>
    </div>
  );
}
