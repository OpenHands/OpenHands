import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  useAppwriteExecutions,
  useAppwriteFunctionVariables,
  useAppwriteFunctions,
  useConversationAppwriteClient,
} from "#/hooks/query/integrations/use-appwrite-resources";
import { APPWRITE_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  CloudAiPromptForm,
  CloudAiRow,
  CloudAiStatus,
  CloudAiSubNav,
  CloudAiToolbar,
  filterByName,
} from "./cloudai-shared";

type FunctionView = "executions" | "secrets";

export function CloudAiFunctionsPanel() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { workspaceId, client } = useConversationAppwriteClient();
  const [functionId, setFunctionId] = useState<string | null>(null);
  const [functionView, setFunctionView] = useState<FunctionView>("executions");
  const [search, setSearch] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [selectedExecutionLogs, setSelectedExecutionLogs] = useState<
    string | null
  >(null);

  const functions = useAppwriteFunctions(workspaceId);
  const executions = useAppwriteExecutions(workspaceId, functionId);
  const functionVariables = useAppwriteFunctionVariables(
    workspaceId,
    functionId,
  );

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: APPWRITE_QUERY_KEYS.all });
  };

  const filteredFunctions = useMemo(
    () => filterByName(functions.data ?? [], search, (fn) => [fn.name, fn.$id]),
    [functions.data, search],
  );

  const filteredVariables = useMemo(
    () =>
      filterByName(functionVariables.data ?? [], search, (variable) => [
        variable.key,
        variable.$id,
      ]),
    [functionVariables.data, search],
  );

  const handleSubmitFunction = async () => {
    if (!client) return;
    try {
      if (editingId) {
        await client.updateFunction(editingId, {
          name: formValues.name,
        });
      } else {
        await client.createFunction({
          functionId: formValues.id || "unique()",
          name: formValues.name,
          runtime: formValues.runtime || "node-18.0",
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

  const handleSubmitVariable = async () => {
    if (!client || !functionId) return;
    try {
      if (editingId) {
        await client.updateFunctionVariable(functionId, editingId, {
          key: formValues.key,
          value: formValues.value,
          secret: true,
        });
      } else {
        await client.createFunctionVariable(functionId, {
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

  if (functionId) {
    const isSecrets = functionView === "secrets";
    return (
      <div
        data-testid={
          isSecrets
            ? "cloudai-function-secrets-panel"
            : "cloudai-executions-panel"
        }
      >
        <CloudAiToolbar
          title={
            isSecrets
              ? t(I18nKey.CLOUDAI$FUNCTION_SECRETS)
              : t(I18nKey.CLOUDAI$EXECUTIONS)
          }
          onBack={() => {
            setFunctionId(null);
            setFunctionView("executions");
            setSelectedExecutionLogs(null);
            setSearch("");
            setShowForm(false);
            setEditingId(null);
          }}
          onRefresh={() =>
            void (isSecrets
              ? functionVariables.refetch()
              : executions.refetch())
          }
          searchValue={isSecrets ? search : undefined}
          onSearchChange={isSecrets ? setSearch : undefined}
          onCreate={
            isSecrets
              ? () => {
                  setEditingId(null);
                  setFormValues({ key: "", value: "" });
                  setShowForm(true);
                }
              : () => {
                  if (!client) return;
                  void (async () => {
                    try {
                      await client.createExecution(functionId);
                      invalidate();
                      displaySuccessToast(t(I18nKey.CLOUDAI$RUN));
                    } catch (error) {
                      displayErrorToast(retrieveAxiosErrorMessage(error));
                    }
                  })();
                }
          }
          createLabel={
            isSecrets ? t(I18nKey.CLOUDAI$CREATE) : t(I18nKey.CLOUDAI$RUN)
          }
        />
        <CloudAiSubNav
          items={[
            {
              id: "executions",
              label: t(I18nKey.CLOUDAI$EXECUTIONS),
            },
            {
              id: "secrets",
              label: t(I18nKey.CLOUDAI$FUNCTION_SECRETS),
            },
          ]}
          activeId={functionView}
          onChange={(id) => {
            setFunctionView(id as FunctionView);
            setSearch("");
            setShowForm(false);
            setEditingId(null);
            setSelectedExecutionLogs(null);
          }}
        />
        {isSecrets && showForm && (
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
            onSubmit={() => void handleSubmitVariable()}
            onCancel={() => {
              setShowForm(false);
              setEditingId(null);
            }}
            submitLabel={
              editingId ? t(I18nKey.CLOUDAI$EDIT) : t(I18nKey.CLOUDAI$CREATE)
            }
          />
        )}
        {isSecrets ? (
          <>
            <CloudAiStatus
              isLoading={functionVariables.isLoading}
              isError={functionVariables.isError}
              empty={
                !functionVariables.isLoading &&
                (functionVariables.data?.length ?? 0) === 0
              }
              noResults={
                !functionVariables.isLoading &&
                (functionVariables.data?.length ?? 0) > 0 &&
                filteredVariables.length === 0
              }
            />
            <div className="flex flex-col gap-2">
              {filteredVariables.map((variable) => (
                <CloudAiRow
                  key={variable.$id}
                  title={variable.key}
                  subtitle={variable.$id}
                  badge={
                    variable.secret ? t(I18nKey.CLOUDAI$SECRETS) : undefined
                  }
                  onEdit={() => {
                    setEditingId(variable.$id);
                    setFormValues({
                      key: variable.key,
                      value: variable.value ?? "",
                    });
                    setShowForm(true);
                  }}
                  onDelete={() => {
                    if (
                      !client ||
                      !window.confirm(t(I18nKey.CLOUDAI$CONFIRM_DELETE))
                    )
                      return;
                    void (async () => {
                      try {
                        await client.deleteFunctionVariable(
                          functionId,
                          variable.$id,
                        );
                        invalidate();
                      } catch (error) {
                        displayErrorToast(retrieveAxiosErrorMessage(error));
                      }
                    })();
                  }}
                />
              ))}
            </div>
          </>
        ) : (
          <>
            <CloudAiStatus
              isLoading={executions.isLoading}
              isError={executions.isError}
              empty={
                !executions.isLoading && (executions.data?.length ?? 0) === 0
              }
            />
            <div className="flex flex-col gap-2">
              {(executions.data ?? []).map((execution) => (
                <CloudAiRow
                  key={execution.$id}
                  title={execution.$id}
                  subtitle={`${execution.status ?? ""} ${execution.responseStatusCode ?? ""}`}
                  extraActions={
                    <BrandButton
                      type="button"
                      variant="secondary"
                      onClick={() =>
                        setSelectedExecutionLogs(
                          [execution.logs, execution.errors]
                            .filter(Boolean)
                            .join("\n") || t(I18nKey.CLOUDAI$EMPTY),
                        )
                      }
                    >
                      {t(I18nKey.CLOUDAI$LOGS)}
                    </BrandButton>
                  }
                />
              ))}
            </div>
            {selectedExecutionLogs && (
              <pre
                className="mt-3 max-h-48 overflow-auto rounded-md border border-[var(--oh-border)] bg-[var(--oh-surface)] p-3 text-xs text-[var(--oh-muted)]"
                data-testid="cloudai-execution-logs"
              >
                {selectedExecutionLogs}
              </pre>
            )}
          </>
        )}
      </div>
    );
  }

  return (
    <div data-testid="cloudai-functions-panel">
      <CloudAiToolbar
        title={t(I18nKey.CLOUDAI$FUNCTIONS)}
        onRefresh={() => void functions.refetch()}
        searchValue={search}
        onSearchChange={setSearch}
        onCreate={() => {
          setEditingId(null);
          setFormValues({ id: "", name: "", runtime: "node-18.0" });
          setShowForm(true);
        }}
      />
      {showForm && (
        <CloudAiPromptForm
          fields={
            editingId
              ? [{ key: "name", label: t(I18nKey.CLOUDAI$NAME) }]
              : [
                  { key: "id", label: t(I18nKey.CLOUDAI$ID) },
                  { key: "name", label: t(I18nKey.CLOUDAI$NAME) },
                  { key: "runtime", label: t(I18nKey.CLOUDAI$RUNTIME) },
                ]
          }
          values={formValues}
          onChange={(key, value) =>
            setFormValues((prev) => ({ ...prev, [key]: value }))
          }
          onSubmit={() => void handleSubmitFunction()}
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
        isLoading={functions.isLoading}
        isError={functions.isError}
        empty={!functions.isLoading && (functions.data?.length ?? 0) === 0}
        noResults={
          !functions.isLoading &&
          (functions.data?.length ?? 0) > 0 &&
          filteredFunctions.length === 0
        }
      />
      <div className="flex flex-col gap-2">
        {filteredFunctions.map((fn) => (
          <CloudAiRow
            key={fn.$id}
            title={fn.name}
            subtitle={fn.$id}
            badge={fn.runtime}
            onOpen={() => {
              setFunctionId(fn.$id);
              setFunctionView("executions");
              setSearch("");
            }}
            onEdit={() => {
              setEditingId(fn.$id);
              setFormValues({ name: fn.name });
              setShowForm(true);
            }}
            onDelete={() => {
              if (!client || !window.confirm(t(I18nKey.CLOUDAI$CONFIRM_DELETE)))
                return;
              void (async () => {
                try {
                  await client.deleteFunction(fn.$id);
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
