import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import GoogleSignInButton from "#/components/features/auth/google-sign-in";
import { I18nKey } from "#/i18n/declaration";

describe("GoogleSignInButton", () => {
    const mockOnCredentialResponse = vi.fn();
    const mockClientId = "test-client-id-12345";

    const renderComponent = (clientId = mockClientId) =>
        render(
            <GoogleSignInButton
                clientId={clientId}
                onCredentialResponse={mockOnCredentialResponse}
            />,
        );

    beforeEach(() => {
        vi.clearAllMocks();
    });

    afterEach(() => {
        // Clean up any global google mock
        if (window.google) {
            delete (window as any).google;
        }
    });

    it("should render the button with Google logo and text", () => {
        renderComponent();

        const button = screen.getByRole("button");
        expect(button).toBeInTheDocument();
        expect(screen.getByText(I18nKey.AUTH$LOGIN_WITH_GOOGLE)).toBeInTheDocument();
    });

    it("should initialize Google SDK and trigger prompt when clicked with SDK loaded", async () => {
        const user = userEvent.setup();

        // Mock Google Identity Services
        const mockInitialize = vi.fn();
        const mockPrompt = vi.fn();

        (window as any).google = {
            accounts: {
                id: {
                    initialize: mockInitialize,
                    prompt: mockPrompt,
                },
            },
        };

        renderComponent();

        const button = screen.getByRole("button");
        await user.click(button);

        // Verify Google SDK was initialized with correct client_id
        expect(mockInitialize).toHaveBeenCalledWith({
            client_id: mockClientId,
            callback: mockOnCredentialResponse,
        });

        // Verify prompt was called
        expect(mockPrompt).toHaveBeenCalled();
    });

    it("should pass the credential response to callback", async () => {
        const user = userEvent.setup();

        const mockCredentialResponse = {
            credential: "test-jwt-token",
            clientId: mockClientId,
        };

        // Mock Google Identity Services with callback capture
        let capturedCallback: ((response: unknown) => void) | undefined;
        const mockInitialize = vi.fn(
            (config: { client_id: string; callback: (response: unknown) => void }) => {
                capturedCallback = config.callback;
            },
        );
        const mockPrompt = vi.fn();

        (window as any).google = {
            accounts: {
                id: {
                    initialize: mockInitialize,
                    prompt: mockPrompt,
                },
            },
        };

        renderComponent();

        const button = screen.getByRole("button");
        await user.click(button);

        // Simulate Google calling the callback with credentials
        if (capturedCallback) {
            capturedCallback(mockCredentialResponse);
        }

        expect(mockOnCredentialResponse).toHaveBeenCalledWith(
            mockCredentialResponse,
        );
    });

    it("should be a button element with type button", () => {
        renderComponent();

        const button = screen.getByRole("button");
        expect(button).toHaveAttribute("type", "button");
    });
});
