import { redirect } from "react-router";

export const clientLoader = () => redirect("/conversations");

export default function IndexRedirect() {
  return null;
}
