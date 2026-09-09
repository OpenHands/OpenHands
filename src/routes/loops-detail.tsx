import { redirect } from "react-router";
import { loopRunPath } from "#/api/loop-service/loop-constants";

export const clientLoader = ({ params }: { params: { runId: string } }) =>
  redirect(loopRunPath(params.runId));

export default function LoopsDetailLegacyRedirect() {
  return null;
}
