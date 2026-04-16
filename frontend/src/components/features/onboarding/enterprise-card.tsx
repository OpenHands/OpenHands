import { Link } from "react-router";
import { Card } from "#/ui/card";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import { FeatureList } from "./feature-list";

interface EnterpriseCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  features: string[];
  onLearnMore: () => void;
  learnMoreLabel: string;
}

export function EnterpriseCard({
  icon,
  title,
  description,
  features,
  onLearnMore,
  learnMoreLabel,
}: EnterpriseCardProps) {
  return (
    <Card
      theme="dark"
      hover="elevated"
      className={cn(
        "w-full md:w-[438px] md:min-h-[371.5px] flex-col p-6 gap-4",
      )}
    >
      <div className={cn("w-10 h-10")}>{icon}</div>
      <Typography.H3 className={cn("text-lg font-semibold text-white")}>
        {title}
      </Typography.H3>
      <Typography.Text className={cn("text-[#8C8C8C]")}>
        {description}
      </Typography.Text>
      <FeatureList features={features} />
      <Link
        to="/information-request"
        onClick={onLearnMore}
        aria-label={`${learnMoreLabel} ${title}`}
        className={cn(
          "mt-2 w-fit px-6 py-2.5 text-sm rounded-sm",
          "bg-[#050505] text-white border border-[#242424]",
          "hover:bg-white hover:text-black transition-colors",
        )}
      >
        {learnMoreLabel}
      </Link>
    </Card>
  );
}
