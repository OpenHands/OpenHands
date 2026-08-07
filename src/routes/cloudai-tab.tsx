import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { CloudAiDatabasesPanel } from "#/components/features/cloudai/cloudai-databases-panel";
import { CloudAiFunctionsPanel } from "#/components/features/cloudai/cloudai-functions-panel";
import { CloudAiSecretsPanel } from "#/components/features/cloudai/cloudai-secrets-panel";
import { CloudAiStoragePanel } from "#/components/features/cloudai/cloudai-storage-panel";

type CloudAiSection = "databases" | "functions" | "secrets" | "storage";

export default function CloudAiTab() {
  const { t } = useTranslation("openhands");
  const [section, setSection] = useState<CloudAiSection>("databases");

  const sections: { id: CloudAiSection; label: I18nKey }[] = [
    { id: "databases", label: I18nKey.CLOUDAI$DATABASES },
    { id: "functions", label: I18nKey.CLOUDAI$FUNCTIONS },
    { id: "secrets", label: I18nKey.CLOUDAI$SECRETS },
    { id: "storage", label: I18nKey.CLOUDAI$STORAGE },
  ];

  return (
    <div
      className="flex h-full min-h-0 flex-col"
      data-testid="cloudai-tab"
    >
      <div className="shrink-0 border-b border-[var(--oh-border)] px-3 pt-3 pb-2">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--oh-muted)]">
          {t(I18nKey.COMMON$CLOUDAI)}
        </p>
        <nav
          className="flex gap-1 rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface)] p-1"
          data-testid="cloudai-section-nav"
        >
          {sections.map((item) => (
            <button
              key={item.id}
              type="button"
              data-testid={`cloudai-section-${item.id}`}
              onClick={() => setSection(item.id)}
              className={cn(
                "flex-1 rounded px-2 py-1.5 text-xs font-medium transition-colors",
                section === item.id
                  ? "bg-[var(--oh-surface-raised)] text-white"
                  : "text-[var(--oh-muted)] hover:text-white",
              )}
            >
              {t(item.label)}
            </button>
          ))}
        </nav>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        {section === "databases" && <CloudAiDatabasesPanel />}
        {section === "functions" && <CloudAiFunctionsPanel />}
        {section === "secrets" && <CloudAiSecretsPanel />}
        {section === "storage" && <CloudAiStoragePanel />}
      </div>
    </div>
  );
}
