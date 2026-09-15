import { useFormatCost } from "#/stores/cost-currency-store";

interface CostTextProps {
  amount: number | null | undefined;
  at?: string | Date | number | null;
  detailed?: boolean;
}

export function CostText({ amount, at, detailed = false }: CostTextProps) {
  const formatCost = useFormatCost();
  return formatCost(amount, at, { detailed });
}
