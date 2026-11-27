import React from "react";

interface SettingsSectionHeaderContextValue {
  setHideSectionHeader: (hide: boolean) => void;
}

const SettingsSectionHeaderContext =
  React.createContext<SettingsSectionHeaderContextValue | null>(null);

export function SettingsSectionHeaderProvider({
  setHideSectionHeader,
  children,
}: React.PropsWithChildren<{
  setHideSectionHeader: (hide: boolean) => void;
}>) {
  const value = React.useMemo(
    () => ({ setHideSectionHeader }),
    [setHideSectionHeader],
  );

  return (
    <SettingsSectionHeaderContext.Provider value={value}>
      {children}
    </SettingsSectionHeaderContext.Provider>
  );
}

export function useSettingsSectionHeader() {
  const context = React.useContext(SettingsSectionHeaderContext);
  return (
    context ?? {
      setHideSectionHeader: () => {},
    }
  );
}
