import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RcaImportControl } from "#/components/features/home/rca-import-control";
import type { RcaContext } from "#/utils/rca-context";

const validRca: RcaContext = {
  summary: "Connection pool exhaustion",
  evidence: ["pool max reached"],
  suspectedComponents: ["db-pool"],
  suspectedFiles: ["src/db/pool.ts"],
  recommendedAction: "Raise the pool limit",
  source: "holmesgpt",
};

const renderControl = (value: RcaContext | null = null, onChange = vi.fn()) => {
  render(<RcaImportControl value={value} onChange={onChange} />);
  return onChange;
};

const openModal = async () => {
  const user = userEvent.setup();
  await user.click(screen.getByTestId("open-rca-import"));
  return user;
};

describe("RcaImportControl", () => {
  it("opens the import modal when the trigger is clicked", async () => {
    renderControl();
    await openModal();

    expect(screen.getByTestId("rca-import-modal")).toBeInTheDocument();
    expect(screen.getByTestId("rca-json-input")).toBeInTheDocument();
  });

  it("applies a valid pasted RCA payload and closes the modal", async () => {
    const onChange = renderControl();
    const user = await openModal();

    fireEvent.change(screen.getByTestId("rca-json-input"), {
      target: { value: JSON.stringify(validRca) },
    });
    await user.click(screen.getByTestId("rca-import-apply"));

    expect(onChange).toHaveBeenCalledWith(validRca);
    expect(screen.queryByTestId("rca-import-modal")).not.toBeInTheDocument();
  });

  it("accepts a snake_case payload from an external system", async () => {
    const onChange = renderControl();
    const user = await openModal();

    fireEvent.change(screen.getByTestId("rca-json-input"), {
      target: {
        value: JSON.stringify({
          summary: "Pool exhausted",
          suspected_files: ["src/db/pool.ts"],
          recommended_action: "Increase pool size",
        }),
      },
    });
    await user.click(screen.getByTestId("rca-import-apply"));

    expect(onChange).toHaveBeenCalledWith({
      summary: "Pool exhausted",
      evidence: [],
      suspectedComponents: [],
      suspectedFiles: ["src/db/pool.ts"],
      recommendedAction: "Increase pool size",
      source: undefined,
    });
  });

  it("shows an error for malformed JSON and keeps the modal open", async () => {
    const onChange = renderControl();
    const user = await openModal();

    fireEvent.change(screen.getByTestId("rca-json-input"), {
      target: { value: "{ not json" },
    });
    await user.click(screen.getByTestId("rca-import-apply"));

    expect(screen.getByTestId("rca-import-error")).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByTestId("rca-import-modal")).toBeInTheDocument();
  });

  it("shows an error for a payload without a summary", async () => {
    const onChange = renderControl();
    const user = await openModal();

    fireEvent.change(screen.getByTestId("rca-json-input"), {
      target: { value: JSON.stringify({ evidence: ["x"] }) },
    });
    await user.click(screen.getByTestId("rca-import-apply"));

    expect(screen.getByTestId("rca-import-error")).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("clears the error once the input is edited", async () => {
    renderControl();
    const user = await openModal();

    fireEvent.change(screen.getByTestId("rca-json-input"), {
      target: { value: "bad" },
    });
    await user.click(screen.getByTestId("rca-import-apply"));
    expect(screen.getByTestId("rca-import-error")).toBeInTheDocument();

    fireEvent.change(screen.getByTestId("rca-json-input"), {
      target: { value: JSON.stringify(validRca) },
    });
    expect(screen.queryByTestId("rca-import-error")).not.toBeInTheDocument();
  });

  it("shows the attached chip and removes the context via its button", async () => {
    const onChange = renderControl(validRca);
    const user = userEvent.setup();

    expect(screen.getByTestId("rca-attached-chip")).toBeInTheDocument();

    await user.click(screen.getByTestId("remove-rca-context"));
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it("prefills the modal with the attached context for editing", async () => {
    renderControl(validRca);
    await openModal();

    expect(screen.getByTestId("rca-json-input")).toHaveValue(
      JSON.stringify(validRca, null, 2),
    );
  });
});
