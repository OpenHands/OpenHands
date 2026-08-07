import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  useAppwriteBuckets,
  useAppwriteFiles,
  useConversationAppwriteClient,
} from "#/hooks/query/integrations/use-appwrite-resources";
import { APPWRITE_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";
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

type FilePreview = {
  name: string;
  url: string;
  mimeType: string;
};

function isImageMime(mimeType: string): boolean {
  return mimeType.startsWith("image/");
}

function isVideoMime(mimeType: string): boolean {
  return mimeType.startsWith("video/");
}

function canEmbedMime(mimeType: string): boolean {
  return (
    mimeType === "application/pdf" ||
    mimeType.startsWith("text/") ||
    mimeType === "application/json"
  );
}

function triggerBlobDownload(url: string, fileName: string): void {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export function CloudAiStoragePanel() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { workspaceId, client } = useConversationAppwriteClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [bucketId, setBucketId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [preview, setPreview] = useState<FilePreview | null>(null);
  const [previewLoadingId, setPreviewLoadingId] = useState<string | null>(null);

  const buckets = useAppwriteBuckets(workspaceId);
  const files = useAppwriteFiles(workspaceId, bucketId);

  const filteredBuckets = useMemo(
    () =>
      filterByName(buckets.data ?? [], search, (bucket) => [
        bucket.name,
        bucket.$id,
      ]),
    [buckets.data, search],
  );

  const filteredFiles = useMemo(
    () =>
      filterByName(files.data ?? [], search, (file) => [file.name, file.$id]),
    [files.data, search],
  );

  useEffect(() => {
    return () => {
      if (preview?.url) {
        URL.revokeObjectURL(preview.url);
      }
    };
  }, [preview?.url]);

  const closePreview = () => {
    setPreview((current) => {
      if (current?.url) {
        URL.revokeObjectURL(current.url);
      }
      return null;
    });
  };

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: APPWRITE_QUERY_KEYS.all });
  };

  const openFilePreview = async (file: {
    $id: string;
    name: string;
    mimeType?: string;
  }) => {
    if (!client || !bucketId) return;
    setPreviewLoadingId(file.$id);
    try {
      const blob = await client.getFileViewBlob(bucketId, file.$id);
      const mimeType = blob.type || file.mimeType || "application/octet-stream";
      const typedBlob =
        blob.type === mimeType ? blob : new Blob([blob], { type: mimeType });
      const objectUrl = URL.createObjectURL(typedBlob);
      setPreview((current) => {
        if (current?.url) {
          URL.revokeObjectURL(current.url);
        }
        return {
          name: file.name,
          url: objectUrl,
          mimeType,
        };
      });
    } catch (error) {
      displayErrorToast(retrieveAxiosErrorMessage(error));
    } finally {
      setPreviewLoadingId(null);
    }
  };

  const handleSubmitBucket = async () => {
    if (!client) return;
    try {
      if (editingId) {
        await client.updateBucket(editingId, {
          name: formValues.name,
        });
      } else {
        await client.createBucket({
          bucketId: formValues.id || "unique()",
          name: formValues.name,
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

  if (bucketId) {
    return (
      <div data-testid="cloudai-files-panel" className="relative">
        <CloudAiToolbar
          title={t(I18nKey.CLOUDAI$FILES)}
          onBack={() => {
            closePreview();
            setBucketId(null);
            setSearch("");
          }}
          onRefresh={() => void files.refetch()}
          searchValue={search}
          onSearchChange={setSearch}
          onCreate={() => fileInputRef.current?.click()}
          createLabel={t(I18nKey.CLOUDAI$UPLOAD)}
        />
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          data-testid="cloudai-file-input"
          onChange={(e) => {
            const input = e.currentTarget;
            const file = input.files?.[0];
            if (!file || !client) return;
            void (async () => {
              try {
                await client.createFile(bucketId, {
                  fileId: "unique()",
                  file,
                });
                invalidate();
                displaySuccessToast(t(I18nKey.CLOUDAI$UPLOAD));
              } catch (error) {
                displayErrorToast(retrieveAxiosErrorMessage(error));
              } finally {
                input.value = "";
              }
            })();
          }}
        />
        {preview && (
          <div
            className="mb-3 flex flex-col gap-2 rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface)] p-3"
            data-testid="cloudai-file-preview"
          >
            <div className="flex flex-wrap items-center gap-2">
              <Typography.Text className="min-w-0 flex-1 truncate text-sm font-semibold text-white">
                {preview.name}
              </Typography.Text>
              <BrandButton
                type="button"
                variant="secondary"
                testId="cloudai-preview-download"
                onClick={() => triggerBlobDownload(preview.url, preview.name)}
              >
                {t(I18nKey.CLOUDAI$DOWNLOAD)}
              </BrandButton>
              <BrandButton
                type="button"
                variant="secondary"
                testId="cloudai-preview-close"
                onClick={closePreview}
              >
                {t(I18nKey.BUTTON$CLOSE)}
              </BrandButton>
            </div>
            {isImageMime(preview.mimeType) ? (
              <img
                src={preview.url}
                alt={preview.name}
                className="max-h-[420px] w-full rounded-md object-contain bg-black/40"
                data-testid="cloudai-preview-image"
              />
            ) : isVideoMime(preview.mimeType) ? (
              // eslint-disable-next-line jsx-a11y/media-has-caption -- user-uploaded preview; no captions available
              <video
                src={preview.url}
                controls
                className="max-h-[420px] w-full rounded-md bg-black/40"
                data-testid="cloudai-preview-video"
              />
            ) : canEmbedMime(preview.mimeType) ? (
              <iframe
                title={preview.name}
                src={preview.url}
                className="h-[420px] w-full rounded-md border border-[var(--oh-border)] bg-white"
                data-testid="cloudai-preview-embed"
              />
            ) : (
              <Typography.Text
                className="py-4 text-sm text-[var(--oh-muted)]"
                testId="cloudai-preview-unsupported"
              >
                {t(I18nKey.CLOUDAI$PREVIEW_UNSUPPORTED)}
              </Typography.Text>
            )}
          </div>
        )}
        <CloudAiStatus
          isLoading={files.isLoading}
          isError={files.isError}
          empty={!files.isLoading && (files.data?.length ?? 0) === 0}
          noResults={
            !files.isLoading &&
            (files.data?.length ?? 0) > 0 &&
            filteredFiles.length === 0
          }
        />
        <div className="flex flex-col gap-2">
          {filteredFiles.map((file) => (
            <CloudAiRow
              key={file.$id}
              title={file.name}
              subtitle={`${file.$id}${file.sizeOriginal ? ` · ${file.sizeOriginal} B` : ""}`}
              extraActions={
                <BrandButton
                  type="button"
                  variant="secondary"
                  testId={`cloudai-view-file-${file.$id}`}
                  isDisabled={previewLoadingId === file.$id}
                  onClick={() => void openFilePreview(file)}
                >
                  {previewLoadingId === file.$id
                    ? t(I18nKey.CLOUDAI$LOADING)
                    : t(I18nKey.COMMON$VIEW)}
                </BrandButton>
              }
              onDelete={() => {
                if (
                  !client ||
                  !window.confirm(t(I18nKey.CLOUDAI$CONFIRM_DELETE))
                )
                  return;
                void (async () => {
                  try {
                    await client.deleteFile(bucketId, file.$id);
                    if (preview?.name === file.name) {
                      closePreview();
                    }
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

  return (
    <div data-testid="cloudai-storage-panel">
      <CloudAiToolbar
        title={t(I18nKey.CLOUDAI$BUCKETS)}
        onRefresh={() => void buckets.refetch()}
        searchValue={search}
        onSearchChange={setSearch}
        onCreate={() => {
          setEditingId(null);
          setFormValues({ id: "", name: "" });
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
                ]
          }
          values={formValues}
          onChange={(key, value) =>
            setFormValues((prev) => ({ ...prev, [key]: value }))
          }
          onSubmit={() => void handleSubmitBucket()}
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
        isLoading={buckets.isLoading}
        isError={buckets.isError}
        empty={!buckets.isLoading && (buckets.data?.length ?? 0) === 0}
        noResults={
          !buckets.isLoading &&
          (buckets.data?.length ?? 0) > 0 &&
          filteredBuckets.length === 0
        }
      />
      <div className="flex flex-col gap-2">
        {filteredBuckets.map((bucket) => (
          <CloudAiRow
            key={bucket.$id}
            title={bucket.name}
            subtitle={bucket.$id}
            onOpen={() => {
              setBucketId(bucket.$id);
              setSearch("");
            }}
            onEdit={() => {
              setEditingId(bucket.$id);
              setFormValues({ name: bucket.name });
              setShowForm(true);
            }}
            onDelete={() => {
              if (!client || !window.confirm(t(I18nKey.CLOUDAI$CONFIRM_DELETE)))
                return;
              void (async () => {
                try {
                  await client.deleteBucket(bucket.$id);
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
