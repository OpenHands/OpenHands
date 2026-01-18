export interface IncidentComponent {
  id: string;
  name: string;
  groupname?: string;
  current_status:
    | "operational"
    | "degraded_performance"
    | "partial_outage"
    | "full_outage";
}

export interface Incident {
  id: string;
  name: string;
  status: "investigating" | "identified" | "monitoring" | "resolved";
  url: string;
  last_update_at: string;
  last_update_message: string;
  current_worst_impact:
    | "degraded_performance"
    | "partial_outage"
    | "full_outage";
  affected_components: IncidentComponent[];
}

export interface Maintenance {
  id: string;
  name: string;
  status:
    | "maintenance_scheduled"
    | "maintenance_in_progress"
    | "maintenance_complete";
  last_update_at: string;
  last_update_message: string;
  url: string;
  affected_components: IncidentComponent[];
  starts_at?: string;
  ends_at?: string;
  started_at?: string;
  scheduled_end_at: string;
}

export interface IncidentStatusResponse {
  ongoing_incidents: Incident[];
  in_progress_maintenances: Maintenance[];
  scheduled_maintenances: Maintenance[];
}
