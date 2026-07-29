import React from "react";

interface SidebarMobileNavContextValue {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

const SidebarMobileNavContext =
  React.createContext<SidebarMobileNavContextValue | null>(null);

export function SidebarMobileNavProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = React.useState(false);

  const open = React.useCallback(() => setIsOpen(true), []);
  const close = React.useCallback(() => setIsOpen(false), []);
  const toggle = React.useCallback(() => setIsOpen((prev) => !prev), []);

  const value = React.useMemo(
    () => ({
      isOpen,
      open,
      close,
      toggle,
    }),
    [isOpen, open, close, toggle],
  );

  return (
    <SidebarMobileNavContext.Provider value={value}>
      {children}
    </SidebarMobileNavContext.Provider>
  );
}

export function useSidebarMobileNav(): SidebarMobileNavContextValue {
  const context = React.useContext(SidebarMobileNavContext);
  if (!context) {
    throw new Error(
      "useSidebarMobileNav must be used within SidebarMobileNavProvider",
    );
  }
  return context;
}
