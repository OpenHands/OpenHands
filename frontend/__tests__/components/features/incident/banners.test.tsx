import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import {
  InProgressMaintenanceBanners,
  OngoingIncidentBanners,
  ScheduledMaintenanceBanners,
} from "#/components/features/incident/banners";
import type { IncidentStatusResponse } from "#/api/option-service/incident.types";

vi.mock("@uidotdev/usehooks", () => ({
  useLocalStorage: vi.fn(() => [null, vi.fn()]),
}));

vi.mock("#/hooks/query/use-incident-status", () => ({
  useIncidentStatus: vi.fn(),
}));

import { useIncidentStatus } from "#/hooks/query/use-incident-status";

const mockUseIncidentStatus = vi.mocked(useIncidentStatus);

const mockData: IncidentStatusResponse = {
  ongoing_incidents: [
    {
      id: "inc-1",
      name: "Ongoing Incident 1",
      status: "investigating",
      url: "https://status.example.com/inc-1",
      last_update_at: "2024-01-15T10:00:00Z",
      last_update_message: "Looking into it.",
      current_worst_impact: "partial_outage",
      affected_components: [],
    },
    {
      id: "inc-2",
      name: "Ongoing Incident 2",
      status: "identified",
      url: "https://status.example.com/inc-2",
      last_update_at: "2024-01-15T11:00:00Z",
      last_update_message: "Issue identified.",
      current_worst_impact: "degraded_performance",
      affected_components: [],
    },
  ],
  in_progress_maintenances: [
    {
      id: "maint-1",
      name: "In Progress Maintenance",
      status: "maintenance_in_progress",
      url: "https://status.example.com/maint-1",
      last_update_at: "2024-01-15T09:00:00Z",
      last_update_message: "Maintenance ongoing.",
      affected_components: [],
      scheduled_end_at: "2024-01-15T12:00:00Z",
    },
  ],
  scheduled_maintenances: [
    {
      id: "maint-2",
      name: "Scheduled Maintenance",
      status: "maintenance_scheduled",
      url: "https://status.example.com/maint-2",
      last_update_at: "2024-01-15T08:00:00Z",
      last_update_message: "Maintenance coming up.",
      affected_components: [],
      scheduled_end_at: "2024-01-16T12:00:00Z",
    },
  ],
};

describe("OngoingIncidentBanners", () => {
  it("renders all ongoing incidents", () => {
    mockUseIncidentStatus.mockReturnValue({ data: mockData } as any);

    render(
      <MemoryRouter>
        <OngoingIncidentBanners />
      </MemoryRouter>,
    );

    expect(screen.getByText("Ongoing Incident 1")).toBeInTheDocument();
    expect(screen.getByText("Ongoing Incident 2")).toBeInTheDocument();
  });

  it("renders nothing when there are no ongoing incidents", () => {
    mockUseIncidentStatus.mockReturnValue({
      data: { ...mockData, ongoing_incidents: [] },
    } as any);

    const { container } = render(
      <MemoryRouter>
        <OngoingIncidentBanners />
      </MemoryRouter>,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when data is undefined", () => {
    mockUseIncidentStatus.mockReturnValue({ data: undefined } as any);

    const { container } = render(
      <MemoryRouter>
        <OngoingIncidentBanners />
      </MemoryRouter>,
    );

    expect(container).toBeEmptyDOMElement();
  });
});

describe("InProgressMaintenanceBanners", () => {
  it("renders all in-progress maintenances", () => {
    mockUseIncidentStatus.mockReturnValue({ data: mockData } as any);

    render(
      <MemoryRouter>
        <InProgressMaintenanceBanners />
      </MemoryRouter>,
    );

    expect(screen.getByText("In Progress Maintenance")).toBeInTheDocument();
  });

  it("renders nothing when there are no in-progress maintenances", () => {
    mockUseIncidentStatus.mockReturnValue({
      data: { ...mockData, in_progress_maintenances: [] },
    } as any);

    const { container } = render(
      <MemoryRouter>
        <InProgressMaintenanceBanners />
      </MemoryRouter>,
    );

    expect(container).toBeEmptyDOMElement();
  });
});

describe("ScheduledMaintenanceBanners", () => {
  it("renders all scheduled maintenances", () => {
    mockUseIncidentStatus.mockReturnValue({ data: mockData } as any);

    render(
      <MemoryRouter>
        <ScheduledMaintenanceBanners />
      </MemoryRouter>,
    );

    expect(screen.getByText("Scheduled Maintenance")).toBeInTheDocument();
  });

  it("renders nothing when there are no scheduled maintenances", () => {
    mockUseIncidentStatus.mockReturnValue({
      data: { ...mockData, scheduled_maintenances: [] },
    } as any);

    const { container } = render(
      <MemoryRouter>
        <ScheduledMaintenanceBanners />
      </MemoryRouter>,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
