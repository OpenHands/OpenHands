import { cn } from "#/utils/utils";
import { Text } from "#/ui/typography";
import type { GitOrg } from "#/types/org";
import { ClaimButton } from "./claim-button";

interface GitOrgRowProps {
  org: GitOrg;
  isLast: boolean;
  onClaim: (id: string) => void;
  onDisconnect: (id: string) => void;
}

export function GitOrgRow({
  org,
  isLast,
  onClaim,
  onDisconnect,
}: GitOrgRowProps) {
  return (
    <div
      data-testid={`org-row-${org.id}`}
      className={cn(
        "flex items-center justify-between px-3 h-[53px]",
        !isLast && "border-b border-[#242424]",
      )}
    >
      <span className="text-sm font-normal leading-5">
        <Text className="text-[#8c8c8c]">{org.provider}/</Text>
        <Text className="text-[#fafafa]">{org.name}</Text>
      </span>
      <ClaimButton org={org} onClaim={onClaim} onDisconnect={onDisconnect} />
    </div>
  );
}
