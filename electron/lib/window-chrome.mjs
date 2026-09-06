/**
 * Native window chrome for the Agent Canvas desktop shell.
 *
 * macOS uses a hidden-inset title bar so the traffic lights overlay the
 * renderer. The frontend draws a matching drag strip (`DesktopTitlebar`,
 * 40px) so those buttons sit in empty chrome instead of on the sidebar.
 * Keep `trafficLightPosition.y` vertically centered in that strip.
 */

export const MAC_TITLEBAR_HEIGHT_PX = 40;
export const MAC_TRAFFIC_LIGHT_POSITION = { x: 16, y: 14 };

export function getMainWindowChrome(platform = process.platform) {
  if (platform === "darwin") {
    return {
      titleBarStyle: "hiddenInset",
      trafficLightPosition: { ...MAC_TRAFFIC_LIGHT_POSITION },
    };
  }
  return { titleBarStyle: "default" };
}
