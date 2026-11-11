import React from "react";

export interface NavigationOptions {
  replace?: boolean;
}

export interface NavigationContextValue {
  currentPath: string;
  conversationId: string | null;
  isNavigating: boolean;
  navigate: (to: string, options?: NavigationOptions) => void;
  /**
   * Live read of the active route, for callbacks that outlive the component
   * that scheduled them (e.g. a mutation settling after its row unmounted).
   * A `currentPath` captured in that closure is stale by then, and
   * `window.location` is wrong when the navigation owner drives routing without
   * updating the document URL (an embedding host). Returns an app-relative path.
   */
  getCurrentPath?: () => string;
}

const noop = () => {};

/**
 * A stable getter that always reads the latest route, for the navigation owner
 * to expose as {@link NavigationContextValue.getCurrentPath}. The getter's
 * identity never changes, so a callback holding it keeps seeing the live path
 * even after the component that captured it unmounted.
 */
export function useCurrentPathGetter(pathname: string): () => string {
  const pathRef = React.useRef(pathname);
  React.useEffect(() => {
    pathRef.current = pathname;
  }, [pathname]);
  return React.useCallback(() => pathRef.current, []);
}

const NavigationContext = React.createContext<NavigationContextValue>({
  currentPath: "/",
  conversationId: null,
  isNavigating: false,
  navigate: noop,
});

interface NavigationProviderProps {
  value: NavigationContextValue;
  children: React.ReactNode;
}

export function NavigationProvider({
  value,
  children,
}: NavigationProviderProps) {
  return (
    <NavigationContext.Provider value={value}>
      {children}
    </NavigationContext.Provider>
  );
}

export function useNavigation() {
  return React.useContext(NavigationContext);
}
