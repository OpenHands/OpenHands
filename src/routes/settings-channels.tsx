import { redirect } from "react-router";
import { CHANNELS_PATH } from "#/api/channel-service/channel-constants";

export const clientLoader = () => redirect(CHANNELS_PATH);

export default function SettingsChannelsRedirect() {
  return null;
}
