import { redirect } from "react-router";
import { SkinService } from "#/api/skin-service";
import HomeScreen from "./home";

/**
 * Index route: when a skin is installed it is the default tab, so `/`
 * redirects to /skin. Otherwise the normal home screen renders.
 */
export const clientLoader = async () => {
  try {
    const status = await SkinService.getStatus();
    if (status.installed) {
      return redirect("/skin");
    }
  } catch {
    // No skin service (dev server / cloud) — fall through to home.
  }
  return null;
};

export default HomeScreen;
