import React from "react";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { useSaveAcpSecrets } from "#/hooks/use-save-acp-secrets";
import {
  getAcpProviderSecrets,
  getAcpCredentialConflicts,
  type ACPProviderSecretField,
} from "#/constants/acp-provider-secrets";
import type { ACPProviderConfig } from "#/api/option-service/option.types";

export interface AcpCredentialForm {
  fields: ACPProviderSecretField[];
  values: Record<string, string>;
  setValue: (name: string, value: string) => void;
  secretExists: (name: string) => boolean;
  hasValueFor: (name: string) => boolean;
  conflicts: Array<[string, string]>;
  isDirty: boolean;
  save: (options?: { silent?: boolean }) => Promise<boolean>;
  reset: () => void;
  isSaving: boolean;
}

/**
 * Manages credential form state for a built-in ACP provider.
 * ``providerConfig`` is the SDK-sourced config used to derive api_key and
 * base_url field names; pass the matching ACPProviderConfig when available.
 * Resets typed values whenever ``providerKey`` changes.
 */
export function useAcpCredentialForm(
  providerKey: string | null | undefined,
  providerConfig?: ACPProviderConfig,
): AcpCredentialForm {
  const { data: existingSecrets } = useSearchSecrets();
  const fields = React.useMemo(
    () => getAcpProviderSecrets(providerKey, providerConfig),
    [providerKey, providerConfig],
  );
  const [values, setValues] = React.useState<Record<string, string>>({});
  const { saveFilled, isSaving } = useSaveAcpSecrets(fields);

  React.useEffect(() => {
    setValues({});
  }, [providerKey]);

  const secretExists = React.useCallback(
    (name: string) => (existingSecrets ?? []).some((s) => s.name === name),
    [existingSecrets],
  );

  const hasValueFor = React.useCallback(
    (name: string) => Boolean(values[name]?.trim()) || secretExists(name),
    [values, secretExists],
  );

  const setValue = React.useCallback(
    (name: string, value: string) =>
      setValues((prev) => ({ ...prev, [name]: value })),
    [],
  );

  const reset = React.useCallback(() => setValues({}), []);

  const conflicts = React.useMemo(
    () => getAcpCredentialConflicts(providerKey, hasValueFor),
    [providerKey, hasValueFor],
  );

  return {
    fields,
    values,
    setValue,
    secretExists,
    hasValueFor,
    conflicts,
    isDirty: fields.some((f) => Boolean(values[f.name]?.trim())),
    save: (options) => saveFilled(values, options),
    reset,
    isSaving,
  };
}
