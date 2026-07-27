import { getStatusIcon } from "#/utils/utils";

interface StatusIconProps {
  status: string;
}

export function StatusIcon({ status }: StatusIconProps) {
  return <span className="text-lg">{getStatusIcon(status)}</span>;
}
