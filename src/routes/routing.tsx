import { redirect } from "react-router";
import { ROUTING_PATH } from "#/api/routing-service/routing-constants";

export const clientLoader = () => redirect(ROUTING_PATH);

export default function RoutingLegacyRedirect() {
  return null;
}
