export interface ConfirmationResponseRequest {
  accept: boolean;
  reason?: string;
}

export interface ConfirmationResponseResponse {
  success: boolean;
}

export type EventSortOrder = "TIMESTAMP" | "TIMESTAMP_DESC";

export interface EventSearchOptions {
  /** Maximum number of events to return per page (server caps at 100). */
  limit?: number;
  /** Page ID for pagination. */
  pageId?: string;
  /** Sort order for the underlying server query. */
  sortOrder?: EventSortOrder;
  /** Filter: event timestamp >= this value (ISO 8601). */
  timestampGte?: string;
  /** Filter: event timestamp < this value (ISO 8601). */
  timestampLt?: string;
  /**
   * Surface unsupported cloud pagination instead of degrading to an empty
   * page. Callers that require a complete result (such as transcript export)
   * must set this so a partial history is never mistaken for exhaustion.
   */
  strictPagination?: boolean;
}

export interface EventSearchPage<TEvent> {
  items: TEvent[];
  next_page_id?: string | null;
}
