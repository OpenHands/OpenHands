import React from "react";

type ComposerDockedContextValue = {
  enabled: boolean;
  minimal: boolean;
};

const ComposerDockedContext = React.createContext<ComposerDockedContextValue>({
  enabled: false,
  minimal: false,
});

export function useComposerDockedMinimal(): boolean {
  return React.useContext(ComposerDockedContext).minimal;
}

export function ComposerDockedProvider({
  enabled,
  minimal = false,
  children,
}: {
  enabled: boolean;
  minimal?: boolean;
  children: React.ReactNode;
}) {
  return (
    <ComposerDockedContext.Provider
      value={{ enabled, minimal: enabled && minimal }}
    >
      {children}
    </ComposerDockedContext.Provider>
  );
}
