import { redirect } from "react-router";
import { GRAPH_PATH } from "#/api/graph-service/graph-constants";

export const clientLoader = () => redirect(GRAPH_PATH);

export default function GraphLegacyRedirect() {
  return null;
}
