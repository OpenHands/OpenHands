import { screen, within } from "@testing-library/react";
import { UserEvent } from "@testing-library/user-event";

export const clickOnEditButton = async (
  user: UserEvent,
  container?: HTMLElement,
) => {
  const wrapper = container ? within(container) : screen;

  const ellipsisButton = wrapper.getByTestId("ellipsis-button");
  await user.click(ellipsisButton);

  const menu = screen.getByTestId("context-menu");
  const editButton = within(menu).getByTestId("edit-button");

  await user.click(editButton);
};
