import { AutomationsDashboard } from "#/components/features/automations/dashboard/automations-dashboard";
import { hasAutomationInterface } from "#/manifests/automation-interface";

/**
 * The page renders the interface manifest's copy, so without an admitted
 * manifest there is nothing to render: a 404, which the layout's error
 * boundary renders.
 */
export const clientLoader = () => {
  if (!hasAutomationInterface()) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }
  return null;
};

export default function AutomationsList() {
  return <AutomationsDashboard />;
}
