import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  useAppwriteVariables,
  useConversationAppwriteClient,
} from "#/hooks/query/integrations/use-appwrite-resources";
import { APPWRITE_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  CloudAiPromptForm,
  CloudAiRow,
  CloudAiStatus,
  CloudAiToolbar,
  filterByName,
} from "./cloudai-shared";

export function CloudAiSecretsPanel() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { workspaceId, client } = useConversationAppwriteClient();
  const variables = useAppwriteVariables(workspaceId);
  const [search, setSearch] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [editingId, setEditingId] = useState<string | null>(null);

  const filtered = useMemo(
    () =>
      filterByName(variables.data ?? [], search, (variable) => [
        variable.key,
        variable.$id,
      ]),
    [variables.data, search],
  );

  const invalidate = () => {
    void queryClient.invalidateQueries({
      queryKey: APPWRITE_QUERY_KEYS.variables,
    });
  };

  const handleSubmit = async () => {
    if (!client) return;
    try {
      if (editingId) {
        await client.updateVariable(editingId, {
          key: formValues.key,
          value: formValues.value,
          secret: true,
        });
      } else {
        await client.createVariable({
          key: formValues.key,
          value: formValues.value,
          secret: true,
        });
      }
      setShowForm(false);
      setEditingId(null);
      setFormValues({});
      invalidate();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$SAVE_SUCCESS));
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    }
  };

  return (
    <div data-testid="cloudai-secrets-panel">
      <CloudAiToolbar
        title={t(I18nKey.CLOUDAI$VARIABLES)}
        onRefresh={() => void variables.refetch()}
        searchValue={search}
        onSearchChange={setSearch}
        onCreate={() => {
          setEditingId(null);
          setFormValues({ key: "", value: "" });
          setShowForm(true);
        }}
      />
      {showForm && (
        <CloudAiPromptForm
          fields={[
            { key: "key", label: t(I18nKey.CLOUDAI$NAME) },
            {
              key: "value",
              label: t(I18nKey.CLOUDAI$VALUE),
              type: "password",
            },
          ]}
          values={formValues}
          onChange={(key, value) =>
            setFormValues((prev) => ({ ...prev, [key]: value }))
          }
          onSubmit={() => void handleSubmit()}
          onCancel={() => {
            setShowForm(false);
            setEditingId(null);
          }}
          submitLabel={
            editingId ? t(I18nKey.CLOUDAI$EDIT) : t(I18nKey.CLOUDAI$CREATE)
          }
        />
      )}
      <CloudAiStatus
        isLoading={variables.isLoading}
        isError={variables.isError}
        empty={!variables.isLoading && (variables.data?.length ?? 0) === 0}
        noResults={
          !variables.isLoading &&
          (variables.data?.length ?? 0) > 0 &&
          filtered.length === 0
        }
      />
      <div className="flex flex-col gap-2">
        {filtered.map((variable) => (
          <CloudAiRow
            key={variable.$id}
            title={variable.key}
            subtitle={variable.$id}
            onEdit={() => {
              setEditingId(variable.$id);
              setFormValues({
                key: variable.key,
                value: variable.value ?? "",
              });
              setShowForm(true);
            }}
            onDelete={() => {
              if (!client || !window.confirm(t(I18nKey.CLOUDAI$CONFIRM_DELETE)))
                return;
              void (async () => {
                try {
                  await client.deleteVariable(variable.$id);
                  invalidate();
                } catch (error) {
                  displayErrorToast(retrieveAxiosErrorMessage(error));
                }
              })();
            }}
          />
        ))}
      </div>
    </div>
  );
}
