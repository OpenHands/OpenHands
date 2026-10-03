import React from "react";

export const MODAL_PORTAL_HOST_ATTRIBUTE = "data-agent-server-ui-portal-host";

type ModalPortalHost = HTMLElement | null | undefined;

export const ModalPortalHostContext =
  React.createContext<ModalPortalHost>(undefined);

export const useModalPortalHost = () =>
  React.useContext(ModalPortalHostContext);
