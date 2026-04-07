import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";
import { cn } from "#/utils/utils";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

interface SandboxStatusIndicatorProps {
  sandboxStatus: V1SandboxStatus;
}

// Map V1SandboxStatus to translation keys
const getSandboxStatusLabel = (status: V1SandboxStatus): string => {
  switch (status) {
    case "RUNNING":
      return "COMMON$RUNNING";
    case "STARTING":
      return "COMMON$STARTING";
    case "STOPPED":
      return "COMMON$STOPPED";
    case "PAUSED":
      return "COMMON$PAUSED";
    case "MISSING":
      return "COMMON$ARCHIVED";
    default:
      return "COMMON$STOPPED";
  }
};

export function SandboxStatusIndicator({
  sandboxStatus,
}: SandboxStatusIndicatorProps) {
  const { t } = useTranslation();

  const sandboxStatusBackgroundColor = useMemo(() => {
    switch (sandboxStatus) {
      case "RUNNING":
        return "bg-[#1FBD53]"; // Running/online - green
      case "STARTING":
        return "bg-[#FFD43B]"; // Busy/starting - yellow
      case "PAUSED":
        return "bg-[#A3A3A3]"; // Paused - grey
      case "STOPPED":
        return "bg-[#3C3C49]"; // Stopped - dark grey
      case "MISSING":
        return "bg-[#A3A3A3]"; // Missing - grey (archived)
      default:
        return "bg-[#3C3C49]"; // Default to grey for unknown states
    }
  }, [sandboxStatus]);

  const statusLabel = t(getSandboxStatusLabel(sandboxStatus));

  return (
    <StyledTooltip
      content={statusLabel}
      placement="right"
      showArrow
      tooltipClassName="bg-[#1a1a1a] text-white text-xs shadow-lg"
    >
      <div
        className={cn("w-1.5 h-1.5 rounded-full", sandboxStatusBackgroundColor)}
      />
    </StyledTooltip>
  );
}
