import React from "react";
import {
  useLocation,
  useMatches,
  useNavigate as useReactRouterNavigate,
  useNavigation as useReactRouterNavigation,
} from "react-router";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";

interface MatchWithParams {
  params?: {
    conversationId?: string;
  };
}

export function ReactRouterNavigationProvider({
  children,
}: React.PropsWithChildren) {
  const { pathname, search } = useLocation();
  const navigate = useReactRouterNavigate();
  const routerNavigation = useReactRouterNavigation();
  const matches = useMatches() as MatchWithParams[];

  const conversationId = React.useMemo(() => {
    for (let index = matches.length - 1; index >= 0; index -= 1) {
      const matchedConversationId = matches[index]?.params?.conversationId;
      if (matchedConversationId) {
        return matchedConversationId;
      }
    }

    return null;
  }, [matches]);

  const value = React.useMemo<NavigationContextValue>(
    () => ({
      currentPath: pathname,
      currentSearch: search,
      conversationId,
      isNavigating: Boolean(routerNavigation.location),
      navigate,
    }),
    [pathname, search, conversationId, routerNavigation.location, navigate],
  );

  return <NavigationProvider value={value}>{children}</NavigationProvider>;
}
