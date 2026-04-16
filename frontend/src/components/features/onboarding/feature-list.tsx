import { Typography } from "#/ui/typography";

interface FeatureListProps {
  features: string[];
}

export function FeatureList({ features }: FeatureListProps) {
  return (
    <ul className="flex flex-col gap-1">
      {features.map((feature, index) => (
        <li key={`feature-${index}`} className="flex items-center gap-2">
          <Typography.Text className="text-[#8C8C8C]">•</Typography.Text>
          <Typography.Text className="text-[#8C8C8C]">
            {feature}
          </Typography.Text>
        </li>
      ))}
    </ul>
  );
}
