import { X } from "lucide-react";

interface ClearButtonProps {
  onClear: () => void;
}

export function ClearButton({ onClear }: ClearButtonProps) {
  return (
    <button
      type="button"
      data-testid="dropdown-clear"
      onClick={onClear}
      // eslint-disable-next-line i18next/no-literal-string -- i18n-free UI primitive; callers supply translated labels
      aria-label="Clear selection"
      className="text-white hover:text-[var(--oh-text-tertiary)]"
    >
      <X size={14} />
    </button>
  );
}
