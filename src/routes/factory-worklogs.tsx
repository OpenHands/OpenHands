import {
  FactoryPage,
  requireAutomationInterface,
} from "#/components/features/factories/factory-page";

export const clientLoader = requireAutomationInterface;

export default function FactoryWorklogs() {
  return <FactoryPage section="worklogs" />;
}
