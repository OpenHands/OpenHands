interface PluginChipProps {
  name: string;
}

export function PluginChip({ name }: PluginChipProps) {
  return (
    <span className="inline-flex items-center rounded-full border border-[var(--oh-border)] bg-[var(--oh-surface-deep)] px-3.5 py-1.5 text-sm text-content">
      {name}
    </span>
  );
}
