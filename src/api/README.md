# API Services Guide

## Overview

Services are the abstraction layer between frontend components and backend APIs. Local agent-server API access should use `@openhands/typescript-client` classes directly, with shared connection options from `src/api/agent-server-client-options.ts` for the active local backend host, session API key, and workspace defaults.

Cloud-specific APIs should use the cloud service modules/proxy helpers instead of local agent-server clients.

Each service is a plain object with async methods.

## Structure

Each service lives in its own directory:

```
src/api/
└── feature-service/
    ├── feature-service.api.ts    # Service methods
    └── feature.types.ts          # Types and interfaces
```

## Creating a Service

Use an object literal with named export. Use object destructuring for parameters to make calls self-documenting. Prefer typed `@openhands/typescript-client` classes over generic HTTP calls. If a needed endpoint is missing, add it to `@openhands/typescript-client` first.

```typescript
// feature-service/feature-service.api.ts
import { FeatureClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { Feature, CreateFeatureParams } from "./feature.types";

export const featureService = {
  getFeature: async ({ id }: { id: string }): Promise<Feature> => {
    return new FeatureClient(getAgentServerClientOptions()).getFeature(id);
  },

  createFeature: async (params: CreateFeatureParams): Promise<Feature> => {
    return new FeatureClient(getAgentServerClientOptions()).createFeature(params);
  },
};
```

### Types

Define app-specific types in a separate file within the same directory when the TypeScript client models are not sufficient:

```typescript
// feature-service/feature.types.ts
export interface Feature {
  id: string;
  name: string;
  description: string;
}

export interface CreateFeatureParams {
  name: string;
  description: string;
}
```

## Usage

> [!IMPORTANT]
> **Don't call services directly in components.** Wrap them in TanStack Query hooks.
>
> Why? TanStack Query provides:
>
> - **Caching** - Avoid redundant network requests
> - **Deduplication** - Multiple components requesting the same data share one request
> - **Loading/error states** - Built-in `isLoading`, `isError`, `data` states
> - **Background refetching** - Data stays fresh automatically
>
> Hooks location:
>
> - `src/hooks/query/` for data fetching (`useQuery`)
> - `src/hooks/mutation/` for writes/updates (`useMutation`)

```typescript
// src/hooks/query/use-feature.ts
import { useQuery } from "@tanstack/react-query";
import { featureService } from "#/api/feature-service/feature-service.api";

export const useFeature = (id: string) => {
  return useQuery({
    queryKey: ["feature", id],
    queryFn: () => featureService.getFeature({ id }),
  });
};
```

## Naming Conventions

| Item         | Convention               | Example                  |
| ------------ | ------------------------ | ------------------------ |
| Directory    | `feature-service/`       | `secrets-service/`       |
| Service file | `feature-service.api.ts` | `secrets-service.api.ts` |
| Types file   | `feature.types.ts`       | `secrets.types.ts`       |
| Export name  | `featureService`         | `secretsService`         |
