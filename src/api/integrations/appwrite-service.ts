import axios, { type AxiosRequestConfig } from "axios";
import { NoBackendAvailableError } from "#/api/agent-server-client-options";
import { getEffectiveLocalBackend } from "#/api/backend-registry/active-store";
import { APPWRITE_WORKSPACE_ID_HEADER } from "#/utils/appwrite-integration-secrets";

/**
 * Canvas-owned AppWrite proxy path. Requests go to ingress/static-server,
 * which resolves the API key server-side from the Secrets store for the
 * workspace identified by {@link APPWRITE_WORKSPACE_ID_HEADER}.
 */
export const APPWRITE_PROXY_BASE = "/api/integrations/appwrite";

export type AppwriteDatabase = {
  $id: string;
  name: string;
  $createdAt?: string;
  $updatedAt?: string;
};

export type AppwriteCollection = {
  $id: string;
  name: string;
  databaseId?: string;
  $permissions?: string[];
  documentSecurity?: boolean;
  $createdAt?: string;
  $updatedAt?: string;
};

export type AppwriteAttribute = {
  key: string;
  type: string;
  status?: string;
  required?: boolean;
  array?: boolean;
  default?: unknown;
  elements?: string[];
};

export type AppwriteDocument = {
  $id: string;
  $createdAt?: string;
  $updatedAt?: string;
  [key: string]: unknown;
};

export type AppwriteFunction = {
  $id: string;
  name: string;
  runtime?: string;
  $createdAt?: string;
  $updatedAt?: string;
};

export type AppwriteExecution = {
  $id: string;
  functionId?: string;
  status?: string;
  responseStatusCode?: number;
  logs?: string;
  errors?: string;
  $createdAt?: string;
};

export type AppwriteVariable = {
  $id: string;
  key: string;
  value?: string;
  secret?: boolean;
};

export type AppwriteFunctionVariable = AppwriteVariable;

export type AppwriteBucket = {
  $id: string;
  name: string;
  $createdAt?: string;
  $updatedAt?: string;
};

export type AppwriteFile = {
  $id: string;
  name: string;
  sizeOriginal?: number;
  mimeType?: string;
  $createdAt?: string;
};

function getListItems<T>(
  data: Record<string, unknown>,
  preferredKeys: string[],
): T[] {
  for (const key of preferredKeys) {
    const value = data[key];
    if (Array.isArray(value)) {
      return value as T[];
    }
  }
  return [];
}

async function appwriteRequest<T>(
  workspaceId: string,
  method: string,
  path: string,
  options: {
    data?: unknown;
    params?: Record<string, string | number | boolean | undefined>;
    headers?: Record<string, string>;
    responseType?: AxiosRequestConfig["responseType"];
  } = {},
): Promise<T> {
  if (!workspaceId.trim()) {
    throw new Error("AppWrite workspace id is required");
  }
  const backend = getEffectiveLocalBackend();
  if (!backend) {
    throw new NoBackendAvailableError();
  }

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${backend.host.replace(/\/+$/, "")}${APPWRITE_PROXY_BASE}${normalizedPath}`;
  const apiKey = backend.apiKey?.trim();

  const response = await axios.request<T>({
    method,
    url,
    data: options.data,
    params: options.params,
    responseType: options.responseType,
    headers: {
      Accept:
        options.responseType === "blob" ? "*/*" : "application/json",
      [APPWRITE_WORKSPACE_ID_HEADER]: workspaceId,
      ...(apiKey ? { "X-Session-API-Key": apiKey } : {}),
      ...options.headers,
    },
  });
  return response.data;
}

/** Bound AppWrite client for a single workspace. */
export type AppwriteClient = ReturnType<typeof AppwriteService.forWorkspace>;

export class AppwriteService {
  static forWorkspace(workspaceId: string) {
    const id = workspaceId.trim();
    const request = <T>(
      method: string,
      path: string,
      options?: Parameters<typeof appwriteRequest<T>>[3],
    ) => appwriteRequest<T>(id, method, path, options);

    return {
      async testConnection(): Promise<void> {
        await request("GET", "/v1/health");
      },

      async listDatabases(): Promise<AppwriteDatabase[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          "/v1/databases",
        );
        return getListItems<AppwriteDatabase>(data, ["databases"]);
      },

      async createDatabase(input: {
        databaseId: string;
        name: string;
      }): Promise<AppwriteDatabase> {
        return request("POST", "/v1/databases", {
          data: { databaseId: input.databaseId, name: input.name },
        });
      },

      async updateDatabase(
        databaseId: string,
        input: { name: string },
      ): Promise<AppwriteDatabase> {
        return request(
          "PUT",
          `/v1/databases/${encodeURIComponent(databaseId)}`,
          { data: { name: input.name } },
        );
      },

      async deleteDatabase(databaseId: string): Promise<void> {
        await request(
          "DELETE",
          `/v1/databases/${encodeURIComponent(databaseId)}`,
        );
      },

      async listCollections(databaseId: string): Promise<AppwriteCollection[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections`,
        );
        return getListItems<AppwriteCollection>(data, ["collections"]);
      },

      async createCollection(
        databaseId: string,
        input: {
          collectionId: string;
          name: string;
          permissions?: string[];
          documentSecurity?: boolean;
        },
      ): Promise<AppwriteCollection> {
        return request(
          "POST",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections`,
          {
            data: {
              collectionId: input.collectionId,
              name: input.name,
              permissions: input.permissions ?? [],
              documentSecurity: input.documentSecurity ?? false,
            },
          },
        );
      },

      async updateCollection(
        databaseId: string,
        collectionId: string,
        input: {
          name: string;
          permissions?: string[];
          documentSecurity?: boolean;
        },
      ): Promise<AppwriteCollection> {
        return request(
          "PUT",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections/${encodeURIComponent(collectionId)}`,
          {
            data: {
              name: input.name,
              ...(input.permissions !== undefined
                ? { permissions: input.permissions }
                : {}),
              ...(input.documentSecurity !== undefined
                ? { documentSecurity: input.documentSecurity }
                : {}),
            },
          },
        );
      },

      async deleteCollection(
        databaseId: string,
        collectionId: string,
      ): Promise<void> {
        await request(
          "DELETE",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections/${encodeURIComponent(collectionId)}`,
        );
      },

      async listAttributes(
        databaseId: string,
        collectionId: string,
      ): Promise<AppwriteAttribute[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections/${encodeURIComponent(collectionId)}/attributes`,
        );
        return getListItems<AppwriteAttribute>(data, ["attributes"]);
      },

      async listDocuments(
        databaseId: string,
        collectionId: string,
      ): Promise<AppwriteDocument[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections/${encodeURIComponent(collectionId)}/documents`,
        );
        return getListItems<AppwriteDocument>(data, ["documents"]);
      },

      async createDocument(
        databaseId: string,
        collectionId: string,
        input: { documentId: string; data: Record<string, unknown> },
      ): Promise<AppwriteDocument> {
        return request(
          "POST",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections/${encodeURIComponent(collectionId)}/documents`,
          {
            data: {
              documentId: input.documentId,
              data: input.data,
              permissions: [],
            },
          },
        );
      },

      async updateDocument(
        databaseId: string,
        collectionId: string,
        documentId: string,
        input: { data: Record<string, unknown> },
      ): Promise<AppwriteDocument> {
        return request(
          "PATCH",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections/${encodeURIComponent(collectionId)}/documents/${encodeURIComponent(documentId)}`,
          { data: { data: input.data } },
        );
      },

      async deleteDocument(
        databaseId: string,
        collectionId: string,
        documentId: string,
      ): Promise<void> {
        await request(
          "DELETE",
          `/v1/databases/${encodeURIComponent(databaseId)}/collections/${encodeURIComponent(collectionId)}/documents/${encodeURIComponent(documentId)}`,
        );
      },

      async listFunctions(): Promise<AppwriteFunction[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          "/v1/functions",
        );
        return getListItems<AppwriteFunction>(data, ["functions"]);
      },

      async createFunction(input: {
        functionId: string;
        name: string;
        runtime: string;
      }): Promise<AppwriteFunction> {
        return request("POST", "/v1/functions", {
          data: {
            functionId: input.functionId,
            name: input.name,
            runtime: input.runtime,
          },
        });
      },

      async updateFunction(
        functionId: string,
        input: { name: string },
      ): Promise<AppwriteFunction> {
        return request(
          "PUT",
          `/v1/functions/${encodeURIComponent(functionId)}`,
          { data: { name: input.name } },
        );
      },

      async deleteFunction(functionId: string): Promise<void> {
        await request(
          "DELETE",
          `/v1/functions/${encodeURIComponent(functionId)}`,
        );
      },

      async listExecutions(functionId: string): Promise<AppwriteExecution[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          `/v1/functions/${encodeURIComponent(functionId)}/executions`,
        );
        return getListItems<AppwriteExecution>(data, ["executions"]);
      },

      async createExecution(
        functionId: string,
        body?: string,
      ): Promise<AppwriteExecution> {
        return request(
          "POST",
          `/v1/functions/${encodeURIComponent(functionId)}/executions`,
          { data: body ? { body } : {} },
        );
      },

      async listFunctionVariables(
        functionId: string,
      ): Promise<AppwriteFunctionVariable[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          `/v1/functions/${encodeURIComponent(functionId)}/variables`,
        );
        return getListItems<AppwriteFunctionVariable>(data, ["variables"]);
      },

      async createFunctionVariable(
        functionId: string,
        input: { key: string; value: string; secret?: boolean },
      ): Promise<AppwriteFunctionVariable> {
        return request(
          "POST",
          `/v1/functions/${encodeURIComponent(functionId)}/variables`,
          {
            data: {
              key: input.key,
              value: input.value,
              secret: input.secret ?? true,
            },
          },
        );
      },

      async updateFunctionVariable(
        functionId: string,
        variableId: string,
        input: { key: string; value: string; secret?: boolean },
      ): Promise<AppwriteFunctionVariable> {
        return request(
          "PUT",
          `/v1/functions/${encodeURIComponent(functionId)}/variables/${encodeURIComponent(variableId)}`,
          {
            data: {
              key: input.key,
              value: input.value,
              secret: input.secret ?? true,
            },
          },
        );
      },

      async deleteFunctionVariable(
        functionId: string,
        variableId: string,
      ): Promise<void> {
        await request(
          "DELETE",
          `/v1/functions/${encodeURIComponent(functionId)}/variables/${encodeURIComponent(variableId)}`,
        );
      },

      async listVariables(): Promise<AppwriteVariable[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          "/v1/project/variables",
        );
        return getListItems<AppwriteVariable>(data, ["variables"]);
      },

      async createVariable(input: {
        key: string;
        value: string;
        secret?: boolean;
      }): Promise<AppwriteVariable> {
        return request("POST", "/v1/project/variables", {
          data: {
            key: input.key,
            value: input.value,
            secret: input.secret ?? true,
          },
        });
      },

      async updateVariable(
        variableId: string,
        input: { key: string; value: string; secret?: boolean },
      ): Promise<AppwriteVariable> {
        return request(
          "PUT",
          `/v1/project/variables/${encodeURIComponent(variableId)}`,
          {
            data: {
              key: input.key,
              value: input.value,
              secret: input.secret ?? true,
            },
          },
        );
      },

      async deleteVariable(variableId: string): Promise<void> {
        await request(
          "DELETE",
          `/v1/project/variables/${encodeURIComponent(variableId)}`,
        );
      },

      async listBuckets(): Promise<AppwriteBucket[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          "/v1/storage/buckets",
        );
        return getListItems<AppwriteBucket>(data, ["buckets"]);
      },

      async createBucket(input: {
        bucketId: string;
        name: string;
      }): Promise<AppwriteBucket> {
        return request("POST", "/v1/storage/buckets", {
          data: {
            bucketId: input.bucketId,
            name: input.name,
            permissions: [],
          },
        });
      },

      async updateBucket(
        bucketId: string,
        input: { name: string },
      ): Promise<AppwriteBucket> {
        return request(
          "PUT",
          `/v1/storage/buckets/${encodeURIComponent(bucketId)}`,
          { data: { name: input.name } },
        );
      },

      async deleteBucket(bucketId: string): Promise<void> {
        await request(
          "DELETE",
          `/v1/storage/buckets/${encodeURIComponent(bucketId)}`,
        );
      },

      async listFiles(bucketId: string): Promise<AppwriteFile[]> {
        const data = await request<Record<string, unknown>>(
          "GET",
          `/v1/storage/buckets/${encodeURIComponent(bucketId)}/files`,
        );
        return getListItems<AppwriteFile>(data, ["files"]);
      },

      async createFile(
        bucketId: string,
        input: { fileId: string; file: File },
      ): Promise<AppwriteFile> {
        const form = new FormData();
        form.append("fileId", input.fileId);
        form.append("file", input.file);
        return request(
          "POST",
          `/v1/storage/buckets/${encodeURIComponent(bucketId)}/files`,
          {
            data: form,
            headers: { "Content-Type": "multipart/form-data" },
          },
        );
      },

      async deleteFile(bucketId: string, fileId: string): Promise<void> {
        await request(
          "DELETE",
          `/v1/storage/buckets/${encodeURIComponent(bucketId)}/files/${encodeURIComponent(fileId)}`,
        );
      },

      /**
       * Fetch file bytes for in-browser preview (AppWrite `/view` endpoint).
       */
      async getFileViewBlob(bucketId: string, fileId: string): Promise<Blob> {
        return request<Blob>(
          "GET",
          `/v1/storage/buckets/${encodeURIComponent(bucketId)}/files/${encodeURIComponent(fileId)}/view`,
          { responseType: "blob" },
        );
      },
    };
  }
}
