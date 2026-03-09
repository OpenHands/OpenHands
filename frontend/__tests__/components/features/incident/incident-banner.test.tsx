import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { MemoryRouter } from "react-router";
import { IncidentBanner } from "#/components/features/incident/incident-banner";
import type { Incident, Maintenance } from "#/api/option-service/incident.types";

const mockSetDismissedIncidents = vi.fn();
let mockDismissedIds: Record<string, boolean> | null = null;

vi.mock("@uidotdev/usehooks", () => ({
  useLocalStorage: vi.fn(() => [mockDismissedIds, mockSetDismissedIncidents]),
}));

const mockIncident: Incident = {
  id: "incident-1",
  name: "API Degradation",
  status: "investigating",
  url: "https://status.example.com/incidents/1",
  last_update_at: "2024-01-15T10:00:00Z",
  last_update_message: "We are investigating the issue.",
  current_worst_impact: "partial_outage",
  affected_components: [],
};

const mockMaintenance: Maintenance = {
  id: "maintenance-1",
  name: "Scheduled DB Maintenance",
  status: "maintenance_scheduled",
  url: "https://status.example.com/maintenances/1",
  last_update_at: "2024-01-15T10:00:00Z",
  last_update_message: "Maintenance will begin shortly.",
  affected_components: [],
  scheduled_end_at: "2024-01-15T12:00:00Z",
};

describe("IncidentBanner", () => {
  beforeEach(() => {
    mockDismissedIds = null;
    mockSetDismissedIncidents.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders the incident name and last update message", () => {
    render(
      <MemoryRouter>
        <IncidentBanner incident={mockIncident} />
      </MemoryRouter>,
    );

    expect(screen.getByText("API Degradation")).toBeInTheDocument();
    expect(
      screen.getByText("We are investigating the issue."),
    ).toBeInTheDocument();
  });

  it("renders a link to the incident URL", () => {
    render(
      <MemoryRouter>
        <IncidentBanner incident={mockIncident} />
      </MemoryRouter>,
    );

    const link = screen.getByRole("link");
    expect(link).toHaveAttribute(
      "href",
      "https://status.example.com/incidents/1",
    );
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders a dismiss button with correct aria-label", () => {
    render(
      <MemoryRouter>
        <IncidentBanner incident={mockIncident} />
      </MemoryRouter>,
    );

    expect(
      screen.getByRole("button", { name: "COMMONT$DISMISS" }),
    ).toBeInTheDocument();
  });

  it("calls setter with updated dismissed state when dismiss is clicked", () => {
    render(
      <MemoryRouter>
        <IncidentBanner incident={mockIncident} />
      </MemoryRouter>,
    );

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "COMMONT$DISMISS" }));
    });

    expect(mockSetDismissedIncidents).toHaveBeenCalledWith({ "incident-1": true });
  });

  it("does not render if incident is already dismissed", () => {
    mockDismissedIds = { "incident-1": true };

    render(
      <MemoryRouter>
        <IncidentBanner incident={mockIncident} />
      </MemoryRouter>,
    );

    expect(screen.queryByText("API Degradation")).not.toBeInTheDocument();
  });

  it("renders a non-dismissed incident even when others are dismissed", () => {
    mockDismissedIds = { "incident-1": true };

    const secondIncident: Incident = {
      ...mockIncident,
      id: "incident-2",
      name: "Database Outage",
    };

    render(
      <MemoryRouter>
        <IncidentBanner incident={secondIncident} />
      </MemoryRouter>,
    );

    expect(screen.getByText("Database Outage")).toBeInTheDocument();
  });

  it("hides the banner after dismiss when setter updates state", () => {
    const { rerender } = render(
      <MemoryRouter>
        <IncidentBanner incident={mockIncident} />
      </MemoryRouter>,
    );

    expect(screen.getByText("API Degradation")).toBeInTheDocument();

    // Simulate the localStorage update by updating the mock
    mockDismissedIds = { "incident-1": true };
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "COMMONT$DISMISS" }));
    });

    rerender(
      <MemoryRouter>
        <IncidentBanner incident={mockIncident} />
      </MemoryRouter>,
    );

    expect(screen.queryByText("API Degradation")).not.toBeInTheDocument();
  });

  it.each([
    ["investigating", "bg-red-500"],
    ["identified", "bg-orange-500"],
    ["monitoring", "bg-yellow-500"],
    ["resolved", "bg-green-500"],
  ] as const)(
    "applies correct color class for status '%s'",
    (status, expectedClass) => {
      const incident: Incident = { ...mockIncident, status };
      const { container } = render(
        <MemoryRouter>
          <IncidentBanner incident={incident} />
        </MemoryRouter>,
      );
      expect(container.firstChild).toHaveClass(expectedClass);
    },
  );

  it.each([
    ["maintenance_scheduled", "bg-blue-500"],
    ["maintenance_in_progress", "bg-blue-500"],
    ["maintenance_complete", "bg-green-500"],
  ] as const)(
    "applies correct color class for maintenance status '%s'",
    (status, expectedClass) => {
      const maintenance: Maintenance = { ...mockMaintenance, status };
      const { container } = render(
        <MemoryRouter>
          <IncidentBanner incident={maintenance} />
        </MemoryRouter>,
      );
      expect(container.firstChild).toHaveClass(expectedClass);
    },
  );

  it("applies fallback gray color for unknown status", () => {
    const incident = { ...mockIncident, status: "unknown_status" as any };
    const { container } = render(
      <MemoryRouter>
        <IncidentBanner incident={incident} />
      </MemoryRouter>,
    );
    expect(container.firstChild).toHaveClass("bg-gray-500");
  });
});
