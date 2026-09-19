import { useEffect, useRef } from "react";
import { useSearchParams } from "react-router";
import {
  CONVERSATION_OVERVIEW_ADD_QUERY_PARAM,
  hasConversationOverviewAddIntent,
} from "#/components/features/conversation/conversation-overview-panel.constants";

export function useConversationOverviewAddIntent(onOpen: () => void): void {
  const [searchParams, setSearchParams] = useSearchParams();
  const onOpenRef = useRef(onOpen);

  useEffect(() => {
    onOpenRef.current = onOpen;
  }, [onOpen]);

  useEffect(() => {
    const search = searchParams.toString();
    if (!hasConversationOverviewAddIntent(search)) {
      return;
    }

    onOpenRef.current();

    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete(CONVERSATION_OVERVIEW_ADD_QUERY_PARAM);
    setSearchParams(nextParams, { replace: true });
  }, [searchParams, setSearchParams]);
}
