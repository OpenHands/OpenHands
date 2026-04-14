import { useQuery } from "@tanstack/react-query";
import OptionService from "#/api/option-service/option-service.api";

export const useSecurityAnalyzers = () =>
  useQuery({
    queryKey: ["security-analyzers"],
    queryFn: OptionService.getSecurityAnalyzers,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
