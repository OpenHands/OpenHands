import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";

interface ClickOutsideTestComponentProps {
  callback: () => void;
}

function ClickOutsideTestComponent({
  callback,
}: ClickOutsideTestComponentProps) {
  const ref = useClickOutsideElement<HTMLDivElement>(callback);

  return (
    <div>
      <div data-testid="inside-element" ref={ref} />
      <div data-testid="outside-element" />
    </div>
  );
}

test("call the callback when the element is clicked outside", async () => {
  const user = userEvent.setup();
  const callback = vi.fn();
  render(<ClickOutsideTestComponent callback={callback} />);

  const insideElement = screen.getByTestId("inside-element");
  const outsideElement = screen.getByTestId("outside-element");

  await user.click(insideElement);
  expect(callback).not.toHaveBeenCalled();

  await user.click(outsideElement);
  expect(callback).toHaveBeenCalled();
});
