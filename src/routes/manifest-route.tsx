import { useLoaderData, useLocation, useNavigate } from "react-router";
import { Route } from "./+types/manifest-route";
import { MANIFEST_REGISTRY } from "#/manifests/manifest-sources";
import { ManifestSetupDialog } from "#/components/features/manifest/manifest-setup-dialog";

/** Where dismissing lands when the dialog was opened by a cold deep link. */
const FALLBACK_PATH = "/";

/**
 * Mount point for every route an extension manifest declares.
 *
 * A manifest owns its own paths, so the host cannot enumerate them at build
 * time — it matches the remaining URL against the registry instead. Anything
 * the registry does not claim is a 404 exactly as before, so this catch-all
 * does not swallow unknown routes.
 */
export const clientLoader = ({ params }: Route.ClientLoaderArgs) => {
  // The splat is already relative to any configured base path.
  const pathname = `/${params["*"] ?? ""}`;
  const manifest = MANIFEST_REGISTRY.findByRoutePath(pathname);

  if (!manifest) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }

  return { manifestId: manifest.id };
};

export default function ManifestRoute() {
  const { manifestId } = useLoaderData<typeof clientLoader>();
  const location = useLocation();
  const navigate = useNavigate();

  const manifest = MANIFEST_REGISTRY.findById(manifestId);
  if (!manifest) return null;

  const handleClose = () => {
    // "default" is the initial history entry, so there is nothing to go back to.
    if (location.key === "default") navigate(FALLBACK_PATH, { replace: true });
    else navigate(-1);
  };

  return <ManifestSetupDialog manifest={manifest} onClose={handleClose} />;
}
