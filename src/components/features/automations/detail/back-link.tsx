import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import {
  automationListPath,
  getInterfaceCopy,
} from "#/manifests/automation-interface";

export function BackLink() {
  return (
    <BackNavButton to={automationListPath()}>
      {getInterfaceCopy().detailBackLabel}
    </BackNavButton>
  );
}
