import { useLoaderData, useLocation, useNavigate } from "react-router";
import { Route } from "./+types/automation-setup-route";
import { automationListPath } from "#/manifests/automation-interface";
import { SETUP_REGISTRY } from "#/manifests/manifest-sources";
import { SetupDialog } from "#/components/features/manifest/manifest-setup-dialog";

/**
 * Setup for one catalog entry.
 *
 * The path is not declared by the manifest: it is the same shape for every
 * entry, so the host derives it from the id. An id no entry claims — or one
 * whose manifest failed admission — is a 404, which the layout's existing
 * error boundary renders.
 */
export const clientLoader = ({ params }: Route.ClientLoaderArgs) => {
  const entry = SETUP_REGISTRY.findById(params.automationId ?? "");

  if (!entry) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }

  return { automationId: entry.id };
};

export default function AutomationSetupRoute() {
  const { automationId } = useLoaderData<typeof clientLoader>();
  const location = useLocation();
  const navigate = useNavigate();

  const entry = SETUP_REGISTRY.findById(automationId);
  if (!entry) return null;

  const handleClose = () => {
    // "default" is the initial history entry, so there is nothing to go back
    // to; dismissing a cold deep link lands on the list instead.
    if (location.key === "default")
      navigate(automationListPath(), { replace: true });
    else navigate(-1);
  };

  return <SetupDialog entry={entry} onClose={handleClose} />;
}
