import { useMutation } from "@tanstack/react-query";
import WorkspacesService, {
  type CloneRepositoryRequest,
  type CloneRepositoryResponse,
} from "#/api/workspaces-service/workspaces-service.api";

export function useCloneRepository() {
  return useMutation<CloneRepositoryResponse, Error, CloneRepositoryRequest>({
    mutationFn: (request) => WorkspacesService.cloneRepository(request),
  });
}
