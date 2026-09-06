import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import LoopService from "#/api/loop-service/loop-service.api";
import type {
  CreateLoopTriggerPayload,
  UpdateLoopTriggerPayload,
} from "#/api/loop-service/loop-types";
import { LOOPS_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useLoopDefinitions() {
  return useQuery({
    queryKey: LOOPS_QUERY_KEYS.definitions(),
    queryFn: () => LoopService.listDefinitions(),
  });
}

export function useLoopRuns(definitionId: string | null) {
  return useQuery({
    queryKey: LOOPS_QUERY_KEYS.runs(definitionId ?? ""),
    queryFn: () => LoopService.listRuns(definitionId!),
    enabled: Boolean(definitionId),
  });
}

export function useLoopRun(runId: string | null) {
  return useQuery({
    queryKey: LOOPS_QUERY_KEYS.run(runId ?? ""),
    queryFn: () => LoopService.getRun(runId!),
    enabled: Boolean(runId),
  });
}

export function useLoopTriggers() {
  return useQuery({
    queryKey: LOOPS_QUERY_KEYS.triggers(),
    queryFn: () => LoopService.listTriggers(),
  });
}

export function useLoopEvents() {
  return useQuery({
    queryKey: LOOPS_QUERY_KEYS.events(),
    queryFn: () => LoopService.listEvents(),
  });
}

function useInvalidateLoops() {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: LOOPS_QUERY_KEYS.all });
  };
}

export function useCreateLoopTrigger() {
  const invalidate = useInvalidateLoops();
  return useMutation({
    mutationFn: (payload: CreateLoopTriggerPayload) =>
      LoopService.createTrigger(payload),
    onSuccess: invalidate,
  });
}

export function useUpdateLoopTrigger() {
  const invalidate = useInvalidateLoops();
  return useMutation({
    mutationFn: ({
      triggerId,
      payload,
    }: {
      triggerId: string;
      payload: UpdateLoopTriggerPayload;
    }) => LoopService.updateTrigger(triggerId, payload),
    onSuccess: invalidate,
  });
}

export function useFireLoopTrigger() {
  const invalidate = useInvalidateLoops();
  return useMutation({
    mutationFn: (triggerId: string) => LoopService.fireTrigger(triggerId),
    onSuccess: invalidate,
  });
}
