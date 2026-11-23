import { screen, within } from "@testing-library/react";
import type userEvent from "@testing-library/user-event";
import { I18nKey } from "#/i18n/declaration";

export const openFactoryAction = async (
  user: ReturnType<typeof userEvent.setup>,
  label: I18nKey,
) => {
  await user.click(
    within(await screen.findByTestId("factory-actions")).getByLabelText(
      I18nKey.AUTOMATIONS$ACTIONS_MENU,
    ),
  );
  await user.click(screen.getByText(label));
};
