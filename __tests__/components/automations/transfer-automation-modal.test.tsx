import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { I18nKey } from "#/i18n/declaration";
import { TransferAutomationModal } from "#/components/features/automations/transfer-automation-modal";
import type { CloudServiceAccount } from "#/api/cloud/types";

const bot: CloudServiceAccount = {
  id: "sa-1",
  org_id: "org-1",
  name: "ci-bot",
  is_disabled: false,
};

function renderModal(
  props: Partial<React.ComponentProps<typeof TransferAutomationModal>> = {},
) {
  const defaults: React.ComponentProps<typeof TransferAutomationModal> = {
    automationName: "Nightly",
    isOpen: true,
    serviceAccounts: [bot],
    currentOwnerId: "owner-1",
    canTransferToSelf: false,
    onConfirm: vi.fn(),
    onCancel: vi.fn(),
  };
  const view = render(<TransferAutomationModal {...defaults} {...props} />);
  return {
    ...view,
    rerenderWith: (next: typeof props) =>
      view.rerender(
        <TransferAutomationModal {...defaults} {...props} {...next} />,
      ),
  };
}

describe("TransferAutomationModal", () => {
  it("explains that no service account is available", () => {
    renderModal({ serviceAccounts: [] });

    expect(
      screen.getByText(I18nKey.AUTOMATIONS$TRANSFER_NO_TARGETS),
    ).toBeInTheDocument();
    expect(screen.getByTestId("transfer-automation-confirm")).toBeDisabled();
  });

  it("explains that no other active service account exists", () => {
    renderModal({
      serviceAccounts: [bot, { ...bot, id: "sa-2", is_disabled: true }],
      currentOwnerId: "sa-1",
    });

    expect(
      screen.getByText(I18nKey.AUTOMATIONS$TRANSFER_NO_TARGETS),
    ).toBeInTheDocument();
  });

  it("shows a loading message while the list loads", () => {
    renderModal({ serviceAccounts: [], isLoading: true });

    expect(
      screen.getByText(I18nKey.AUTOMATIONS$TRANSFER_LOADING),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(I18nKey.AUTOMATIONS$TRANSFER_NO_TARGETS),
    ).not.toBeInTheDocument();
  });

  it("shows an error message when the list fails to load", () => {
    renderModal({ serviceAccounts: [], isError: true });

    expect(
      screen.getByText(I18nKey.AUTOMATIONS$TRANSFER_LOAD_ERROR),
    ).toBeInTheDocument();
  });

  it("selects the first service account when the list arrives after the modal opened", () => {
    const { rerenderWith } = renderModal({
      serviceAccounts: [],
      isLoading: true,
    });

    rerenderWith({ serviceAccounts: [bot], isLoading: false });

    expect(screen.getByTestId("transfer-automation-target")).toHaveValue(
      "sa-1",
    );
    expect(screen.getByTestId("transfer-automation-confirm")).toBeEnabled();
  });
});
