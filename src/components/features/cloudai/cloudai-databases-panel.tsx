import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  useAppwriteAttributes,
  useAppwriteCollections,
  useAppwriteDatabases,
  useAppwriteDocuments,
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
  buildDocumentFormFields,
  buildDocumentFormValues,
  documentListSubtitle,
  documentListTitle,
  formValuesToDocumentData,
} from "./cloudai-document-form";
import {
  CloudAiPromptForm,
  CloudAiRow,
  CloudAiStatus,
  CloudAiToolbar,
  filterByName,
} from "./cloudai-shared";

type Level = "databases" | "collections" | "documents" | "security";

function permissionsToText(permissions: string[] | undefined): string {
  return (permissions ?? []).join("\n");
}

function textToPermissions(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export function CloudAiDatabasesPanel() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { workspaceId, client } = useConversationAppwriteClient();
  const [level, setLevel] = useState<Level>("databases");
  const [databaseId, setDatabaseId] = useState<string | null>(null);
  const [collectionId, setCollectionId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [securityForm, setSecurityForm] = useState({
    permissions: "",
    documentSecurity: false,
  });

  const databases = useAppwriteDatabases(workspaceId);
  const collections = useAppwriteCollections(workspaceId, databaseId);
  const documents = useAppwriteDocuments(workspaceId, databaseId, collectionId);
  const attributes = useAppwriteAttributes(
    workspaceId,
    level === "documents" ? databaseId : null,
    level === "documents" ? collectionId : null,
  );

  const documentFormLabels = useMemo(
    () => ({
      id: t(I18nKey.CLOUDAI$ID),
      arrayHelp: t(I18nKey.CLOUDAI$ARRAY_JSON_HELP),
      datetimeHelp: t(I18nKey.CLOUDAI$DATETIME_HELP),
      jsonFallback: t(I18nKey.CLOUDAI$DATA),
    }),
    [t],
  );

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: APPWRITE_QUERY_KEYS.all });
  };

  const openCollections = (id: string) => {
    setDatabaseId(id);
    setCollectionId(null);
    setLevel("collections");
    setSearch("");
    setShowForm(false);
  };

  const openDocuments = (id: string) => {
    setCollectionId(id);
    setLevel("documents");
    setSearch("");
    setShowForm(false);
  };

  const openSecurity = (id: string) => {
    const collection = collections.data?.find((c) => c.$id === id);
    setCollectionId(id);
    setSecurityForm({
      permissions: permissionsToText(collection?.$permissions),
      documentSecurity: Boolean(collection?.documentSecurity),
    });
    setLevel("security");
    setSearch("");
    setShowForm(false);
  };

  const handleBack = () => {
    setShowForm(false);
    setEditingId(null);
    setSearch("");
    if (level === "security") {
      setLevel("collections");
      setCollectionId(null);
      return;
    }
    if (level === "documents") {
      setCollectionId(null);
      setLevel("collections");
      return;
    }
    if (level === "collections") {
      setDatabaseId(null);
      setLevel("databases");
    }
  };

  const handleDelete = async (kind: Level, id: string) => {
    if (!client || !window.confirm(t(I18nKey.CLOUDAI$CONFIRM_DELETE))) return;
    try {
      if (kind === "databases") {
        await client.deleteDatabase(id);
      } else if (kind === "collections" && databaseId) {
        await client.deleteCollection(databaseId, id);
      } else if (kind === "documents" && databaseId && collectionId) {
        await client.deleteDocument(databaseId, collectionId, id);
      }
      invalidate();
      displaySuccessToast(t(I18nKey.CLOUDAI$DELETE));
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    }
  };

  const handleSubmit = async () => {
    if (!client) return;
    try {
      if (level === "databases") {
        if (editingId) {
          await client.updateDatabase(editingId, {
            name: formValues.name,
          });
        } else {
          await client.createDatabase({
            databaseId: formValues.id || "unique()",
            name: formValues.name,
          });
        }
      } else if (level === "collections" && databaseId) {
        if (editingId) {
          await client.updateCollection(databaseId, editingId, {
            name: formValues.name,
          });
        } else {
          await client.createCollection(databaseId, {
            collectionId: formValues.id || "unique()",
            name: formValues.name,
          });
        }
      } else if (level === "documents" && databaseId && collectionId) {
        const parsed = formValuesToDocumentData(formValues, attributes.data);
        if (editingId) {
          await client.updateDocument(databaseId, collectionId, editingId, {
            data: parsed,
          });
        } else {
          await client.createDocument(databaseId, collectionId, {
            documentId: formValues.id || "unique()",
            data: parsed,
          });
        }
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

  const handleSaveSecurity = async () => {
    if (!client || !databaseId || !collectionId) return;
    const collection = collections.data?.find((c) => c.$id === collectionId);
    if (!collection) return;
    try {
      await client.updateCollection(databaseId, collectionId, {
        name: collection.name,
        permissions: textToPermissions(securityForm.permissions),
        documentSecurity: securityForm.documentSecurity,
      });
      invalidate();
      displaySuccessToast(t(I18nKey.INTEGRATIONS$SAVE_SUCCESS));
      setLevel("collections");
      setCollectionId(null);
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    }
  };

  const title =
    level === "databases"
      ? t(I18nKey.CLOUDAI$DATABASES)
      : level === "collections"
        ? t(I18nKey.CLOUDAI$COLLECTIONS)
        : level === "security"
          ? t(I18nKey.CLOUDAI$SECURITY_RULES)
          : t(I18nKey.CLOUDAI$DOCUMENTS);

  const activeQuery =
    level === "databases"
      ? databases
      : level === "collections" || level === "security"
        ? collections
        : documents;

  const databaseItems = useMemo(
    () => filterByName(databases.data ?? [], search, (d) => [d.name, d.$id]),
    [databases.data, search],
  );
  const collectionItems = useMemo(
    () => filterByName(collections.data ?? [], search, (c) => [c.name, c.$id]),
    [collections.data, search],
  );
  const documentItems = useMemo(
    () =>
      filterByName(documents.data ?? [], search, (d) => [
        d.$id,
        documentListTitle(d),
        documentListSubtitle(d),
        JSON.stringify(d),
      ]),
    [documents.data, search],
  );

  const items =
    level === "databases"
      ? databaseItems.map((d) => ({
          id: d.$id,
          title: d.name,
          subtitle: d.$id,
        }))
      : level === "collections"
        ? collectionItems.map((c) => ({
            id: c.$id,
            title: c.name,
            subtitle: c.$id,
          }))
        : documentItems.map((d) => ({
            id: d.$id,
            title: documentListTitle(d),
            subtitle: documentListSubtitle(d),
          }));

  const sourceCount =
    level === "databases"
      ? (databases.data?.length ?? 0)
      : level === "collections"
        ? (collections.data?.length ?? 0)
        : (documents.data?.length ?? 0);

  const editingDocument = useMemo(() => {
    if (!editingId || level !== "documents") return undefined;
    return documents.data?.find((d) => d.$id === editingId);
  }, [documents.data, editingId, level]);

  const formFields =
    level === "documents"
      ? buildDocumentFormFields(
          attributes.data,
          editingDocument,
          documentFormLabels,
          !editingId,
        )
      : [
          ...(editingId ? [] : [{ key: "id", label: t(I18nKey.CLOUDAI$ID) }]),
          { key: "name", label: t(I18nKey.CLOUDAI$NAME) },
        ];

  if (level === "security") {
    return (
      <div data-testid="cloudai-security-panel">
        <CloudAiToolbar
          title={title}
          onBack={handleBack}
          onRefresh={() => void collections.refetch()}
        />
        <div className="mb-3 flex flex-col gap-3 rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface)] p-3">
          <label className="flex items-center gap-2 text-sm text-white">
            <input
              type="checkbox"
              data-testid="cloudai-document-security"
              checked={securityForm.documentSecurity}
              onChange={(e) =>
                setSecurityForm((prev) => ({
                  ...prev,
                  documentSecurity: e.target.checked,
                }))
              }
              className="size-4"
            />
            {t(I18nKey.CLOUDAI$DOCUMENT_SECURITY)}
          </label>
          <label className="flex flex-col gap-1 text-xs text-[var(--oh-muted)]">
            {t(I18nKey.CLOUDAI$PERMISSIONS)}
            <textarea
              data-testid="cloudai-permissions"
              className="min-h-32 rounded-md border border-[var(--oh-border)] bg-transparent p-2 font-mono text-sm text-white"
              value={securityForm.permissions}
              onChange={(e) =>
                setSecurityForm((prev) => ({
                  ...prev,
                  permissions: e.target.value,
                }))
              }
            />
            <span className="text-[11px]">
              {t(I18nKey.CLOUDAI$PERMISSIONS_HELP)}
            </span>
          </label>
          <div className="flex gap-2">
            <BrandButton
              type="button"
              variant="primary"
              testId="cloudai-save-security"
              onClick={() => void handleSaveSecurity()}
            >
              {t(I18nKey.CLOUDAI$SAVE_RULES)}
            </BrandButton>
            <BrandButton type="button" variant="secondary" onClick={handleBack}>
              {t(I18nKey.BUTTON$CANCEL)}
            </BrandButton>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div data-testid="cloudai-databases-panel">
      <CloudAiToolbar
        title={title}
        onBack={level !== "databases" ? handleBack : undefined}
        onRefresh={() => void activeQuery.refetch()}
        searchValue={search}
        onSearchChange={setSearch}
        onCreate={() => {
          setEditingId(null);
          setFormValues(
            level === "documents"
              ? buildDocumentFormValues(attributes.data, undefined, true)
              : { id: "", name: "" },
          );
          setShowForm(true);
        }}
      />
      {showForm && (
        <CloudAiPromptForm
          fields={formFields}
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
        isLoading={activeQuery.isLoading}
        isError={activeQuery.isError}
        empty={!activeQuery.isLoading && sourceCount === 0}
        noResults={
          !activeQuery.isLoading && sourceCount > 0 && items.length === 0
        }
      />
      <div className="flex flex-col gap-2">
        {items.map((item) => (
          <CloudAiRow
            key={item.id}
            title={item.title}
            subtitle={item.subtitle}
            onOpen={
              level === "databases"
                ? () => openCollections(item.id)
                : level === "collections"
                  ? () => openDocuments(item.id)
                  : undefined
            }
            extraActions={
              level === "collections" ? (
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={() => openSecurity(item.id)}
                >
                  {t(I18nKey.CLOUDAI$SECURITY)}
                </BrandButton>
              ) : undefined
            }
            onEdit={() => {
              setEditingId(item.id);
              if (level === "documents") {
                const doc = documents.data?.find((d) => d.$id === item.id);
                setFormValues(
                  buildDocumentFormValues(attributes.data, doc, false),
                );
              } else {
                setFormValues({ name: item.title });
              }
              setShowForm(true);
            }}
            onDelete={() => void handleDelete(level, item.id)}
          />
        ))}
      </div>
    </div>
  );
}
