import { describe, expect, it } from "vitest";
import { HttpError } from "@openhands/typescript-client";
import { extractDraftDispatchErrors } from "#/components/features/automations/setup/automation-setup-draft-service";

const detailEnvelope = {
  detail: {
    message: "Draft is not dispatchable",
    errors: [
      {
        field: "prompt",
        code: "required",
        message: "Prompt is required",
      },
    ],
  },
};

describe("extractDraftDispatchErrors", () => {
  it("reads FastAPI detail envelopes from axios draft dispatch errors", () => {
    const error = {
      isAxiosError: true,
      response: { data: detailEnvelope },
    };

    expect(extractDraftDispatchErrors(error)).toBe("Prompt is required");
  });

  it("reads FastAPI detail envelopes from SDK HttpError draft dispatch errors", () => {
    const error = new HttpError(422, "Unprocessable Entity", detailEnvelope);

    expect(extractDraftDispatchErrors(error)).toBe("Prompt is required");
  });

  it("falls back to the detail message when no field errors are present", () => {
    const error = new HttpError(422, "Unprocessable Entity", {
      detail: { message: "Draft is not dispatchable", errors: [] },
    });

    expect(extractDraftDispatchErrors(error)).toBe("Draft is not dispatchable");
  });
});
