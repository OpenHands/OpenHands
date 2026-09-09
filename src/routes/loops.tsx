import { redirect } from "react-router";
import { LOOPS_PATH } from "#/api/loop-service/loop-constants";

export const clientLoader = () => redirect(LOOPS_PATH);

export default function LoopsLegacyRedirect() {
  return null;
}
