import { Box } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { SandboxTemplateOption } from "#/api/cloud/sandbox-template-service.api";
import { cn } from "#/utils/utils";
import {
  formControlBorderClassName,
  formControlSurfaceClassName,
} from "#/utils/form-control-classes";

interface SandboxSelectorProps {
  items: SandboxTemplateOption[];
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
}

export function SandboxSelector({
  items,
  value,
  onChange,
  disabled,
}: SandboxSelectorProps) {
  const { t } = useTranslation("openhands");
  return (
    <label
      className={cn(
        "flex min-w-0 max-w-full items-center gap-2 rounded-full px-2.5 py-1 text-sm text-white focus-within:ring-2 focus-within:ring-white/40",
        formControlBorderClassName,
        formControlSurfaceClassName,
        disabled && "opacity-50",
      )}
    >
      <Box size={16} className="shrink-0" aria-hidden />
      <span>{t(I18nKey.HOME$SANDBOX_LABEL)}</span>
      <select
        data-testid="sandbox-selector"
        aria-label={t(I18nKey.HOME$SANDBOX_LABEL)}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        className="min-w-0 max-w-48 cursor-pointer truncate bg-transparent outline-none disabled:cursor-not-allowed"
      >
        <option value="" className="bg-neutral-900">
          {t(I18nKey.HOME$SANDBOX_DEFAULT)}
        </option>
        {items.map((item) => (
          <option key={item.id} value={item.id} className="bg-neutral-900">
            {item.display_name}
          </option>
        ))}
      </select>
    </label>
  );
}
