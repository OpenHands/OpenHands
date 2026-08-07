import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { AppwriteService } from "#/api/integrations/appwrite-service";
import { APPWRITE_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  useAppwriteIntegration,
  useConversationAppwriteIntegration,
} from "#/hooks/query/use-appwrite-integration";

function useWorkspaceAppwriteClient(workspaceId: string | null) {
  const { isReady } = useAppwriteIntegration(workspaceId);
  const client = useMemo(
    () => (workspaceId ? AppwriteService.forWorkspace(workspaceId) : null),
    [workspaceId],
  );
  return { client, isReady: isReady && Boolean(client) };
}

export function useConversationAppwriteClient() {
  const integration = useConversationAppwriteIntegration();
  const client = useMemo(
    () =>
      integration.workspaceId
        ? AppwriteService.forWorkspace(integration.workspaceId)
        : null,
    [integration.workspaceId],
  );
  return {
    ...integration,
    client,
    isReady: integration.isReady && Boolean(client),
  };
}

export function useAppwriteDatabases(workspaceId: string | null) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [...APPWRITE_QUERY_KEYS.databases, workspaceId ?? ""],
    queryFn: () => client!.listDatabases(),
    enabled: isReady,
  });
}

export function useAppwriteCollections(
  workspaceId: string | null,
  databaseId: string | null,
) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [
      ...APPWRITE_QUERY_KEYS.collections(databaseId ?? ""),
      workspaceId ?? "",
    ],
    queryFn: () => client!.listCollections(databaseId!),
    enabled: isReady && Boolean(databaseId),
  });
}

export function useAppwriteDocuments(
  workspaceId: string | null,
  databaseId: string | null,
  collectionId: string | null,
) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [
      ...APPWRITE_QUERY_KEYS.documents(databaseId ?? "", collectionId ?? ""),
      workspaceId ?? "",
    ],
    queryFn: () => client!.listDocuments(databaseId!, collectionId!),
    enabled: isReady && Boolean(databaseId) && Boolean(collectionId),
  });
}

export function useAppwriteAttributes(
  workspaceId: string | null,
  databaseId: string | null,
  collectionId: string | null,
) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [
      ...APPWRITE_QUERY_KEYS.attributes(databaseId ?? "", collectionId ?? ""),
      workspaceId ?? "",
    ],
    queryFn: () => client!.listAttributes(databaseId!, collectionId!),
    enabled: isReady && Boolean(databaseId) && Boolean(collectionId),
  });
}

export function useAppwriteFunctions(workspaceId: string | null) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [...APPWRITE_QUERY_KEYS.functions, workspaceId ?? ""],
    queryFn: () => client!.listFunctions(),
    enabled: isReady,
  });
}

export function useAppwriteExecutions(
  workspaceId: string | null,
  functionId: string | null,
) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [
      ...APPWRITE_QUERY_KEYS.executions(functionId ?? ""),
      workspaceId ?? "",
    ],
    queryFn: () => client!.listExecutions(functionId!),
    enabled: isReady && Boolean(functionId),
  });
}

export function useAppwriteFunctionVariables(
  workspaceId: string | null,
  functionId: string | null,
) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [
      ...APPWRITE_QUERY_KEYS.functionVariables(functionId ?? ""),
      workspaceId ?? "",
    ],
    queryFn: () => client!.listFunctionVariables(functionId!),
    enabled: isReady && Boolean(functionId),
  });
}

export function useAppwriteVariables(workspaceId: string | null) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [...APPWRITE_QUERY_KEYS.variables, workspaceId ?? ""],
    queryFn: () => client!.listVariables(),
    enabled: isReady,
  });
}

export function useAppwriteBuckets(workspaceId: string | null) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [...APPWRITE_QUERY_KEYS.buckets, workspaceId ?? ""],
    queryFn: () => client!.listBuckets(),
    enabled: isReady,
  });
}

export function useAppwriteFiles(
  workspaceId: string | null,
  bucketId: string | null,
) {
  const { client, isReady } = useWorkspaceAppwriteClient(workspaceId);
  return useQuery({
    queryKey: [...APPWRITE_QUERY_KEYS.files(bucketId ?? ""), workspaceId ?? ""],
    queryFn: () => client!.listFiles(bucketId!),
    enabled: isReady && Boolean(bucketId),
  });
}
