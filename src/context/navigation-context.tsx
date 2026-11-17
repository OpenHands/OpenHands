import React from "react";

export interface NavigationOptions {
  replace?: boolean;
}

export interface NavigationContextValue {
  currentPath: string;
  /** The URL's query string, with its leading `?` when not empty. */
  currentSearch?: string;
  conversationId: string | null;
  isNavigating: boolean;
  navigate: (to: string, options?: NavigationOptions) => void;
}

const noop = () => {};

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
