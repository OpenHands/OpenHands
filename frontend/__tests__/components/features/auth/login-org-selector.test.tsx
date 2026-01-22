import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import LoginOrgSelector from "#/components/features/auth/otp-login/login-org-selector";
import { I18nKey } from "#/i18n/declaration";

describe("LoginOrgSelector", () => {
    const mockSetIsOrgSelected = vi.fn();

    const renderComponent = () =>
        render(<LoginOrgSelector setIsOrgSelected={mockSetIsOrgSelected} />);

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render heading", () => {
        renderComponent();

        expect(screen.getByText(I18nKey.AUTH$LOGIN_AS)).toBeInTheDocument();
    });

    // it("should render organization options", () => {
    //     renderComponent();

    //     // Check for org buttons (using the hardcoded org1 and org2)
    //     expect(screen.getByRole("button", { name: "org1" })).toBeInTheDocument();
    //     expect(screen.getByRole("button", { name: "org2" })).toBeInTheDocument();
    // });

    it("should render remember checkbox and login button", () => {
        renderComponent();

        expect(screen.getByText(I18nKey.AUTH$ALWAYS_LOGIN_TO_LAST_USED)).toBeInTheDocument();
        expect(screen.getByRole("checkbox")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: I18nKey.AUTH$LOGIN })).toBeInTheDocument();
    });

    it("should have disabled login button when no org is selected", () => {
        renderComponent();

        const loginButton = screen.getByRole("button", { name: I18nKey.AUTH$LOGIN });
        expect(loginButton).toBeDisabled();
    });

    it("should enable login button when an org is selected", async () => {
        const user = userEvent.setup();
        renderComponent();

        const org1Button = screen.getByRole("button", { name: "org1" });
        await user.click(org1Button);

        const loginButton = screen.getByRole("button", { name: I18nKey.AUTH$LOGIN });
        expect(loginButton).not.toBeDisabled();
    });

    it("should highlight selected org", async () => {
        const user = userEvent.setup();
        renderComponent();

        const org1Button = screen.getByRole("button", { name: "org1" });
        await user.click(org1Button);

        // The selected org should have border-white class
        expect(org1Button).toHaveClass("border-white");
    });

    it("should call setIsOrgSelected(true) when login button is clicked with selected org", async () => {
        const user = userEvent.setup();
        renderComponent();

        // Select an org first
        const org1Button = screen.getByRole("button", { name: "org1" });
        await user.click(org1Button);

        // Click login
        const loginButton = screen.getByRole("button", { name: I18nKey.AUTH$LOGIN });
        await user.click(loginButton);

        expect(mockSetIsOrgSelected).toHaveBeenCalledWith(true);
    });

    it("should not call setIsOrgSelected when login is clicked without org selection", async () => {
        const user = userEvent.setup();
        renderComponent();

        const loginButton = screen.getByRole("button", { name: I18nKey.AUTH$LOGIN });
        // Button is disabled, but try to click anyway
        await user.click(loginButton);

        expect(mockSetIsOrgSelected).not.toHaveBeenCalled();
    });

    it("should allow checking remember checkbox", async () => {
        const user = userEvent.setup();
        renderComponent();

        const checkbox = screen.getByRole("checkbox");
        expect(checkbox).not.toBeChecked();

        await user.click(checkbox);
        expect(checkbox).toBeChecked();
    });
});
