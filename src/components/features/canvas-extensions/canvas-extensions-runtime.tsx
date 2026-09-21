import React from "react";
import { useNavigate } from "react-router";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { loadCanvasExtensionModule } from "#/extensions/canvas-extension-module-loader";
import { useCanvasExtensions } from "#/hooks/query/use-canvas-extensions";
import {
  CANVAS_EXTENSION_HOST_API_VERSION,
  type CanvasExtensionDispose,
  type CanvasExtensionHost,
  type CanvasExtensionModule,
  type CanvasExtensionPageContribution,
  type CanvasExtensionPageMount,
  type InstalledCanvasExtensionInfo,
} from "#/types/canvas-extension";

export interface RegisteredCanvasExtensionPage {
  extension: InstalledCanvasExtensionInfo;
  contribution: CanvasExtensionPageContribution;
  mount: CanvasExtensionPageMount;
  href: string;
}

interface CanvasExtensionsRuntimeValue {
  pages: RegisteredCanvasExtensionPage[];
  activating: boolean;
  errors: ReadonlyMap<string, string>;
}

const EMPTY_RUNTIME: CanvasExtensionsRuntimeValue = {
  pages: [],
  activating: false,
  errors: new Map(),
};

const CanvasExtensionsRuntimeContext =
  React.createContext<CanvasExtensionsRuntimeValue>(EMPTY_RUNTIME);

export function useCanvasExtensionsRuntime(): CanvasExtensionsRuntimeValue {
  return React.useContext(CanvasExtensionsRuntimeContext);
}

function isValidSegment(value: string): boolean {
  return /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value);
}

export function buildCanvasExtensionPageHref(
  extensionName: string,
  contributionPath: string,
): string {
  return `/extensions/${encodeURIComponent(extensionName)}/${contributionPath
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;
}

function getDeclaredPage(
  extension: InstalledCanvasExtensionInfo,
  contributionId: string,
): CanvasExtensionPageContribution {
  const contribution = extension.manifest?.contributes?.pages?.find(
    (page) => page.id === contributionId,
  );
  if (!contribution) {
    throw new Error(
      `Extension ${extension.name} registered undeclared page "${contributionId}".`,
    );
  }
  // The backend declares page paths as absolute routes (e.g. "/dashboard");
  // normalize to the relative form used for hrefs and route matching.
  const normalizedPath = contribution.path.replace(/^\/+/, "");
  if (
    !isValidSegment(extension.name) ||
    !isValidSegment(contribution.id) ||
    !normalizedPath.split("/").every(isValidSegment)
  ) {
    throw new Error(
      `Extension ${extension.name} has an invalid page name, id, or path.`,
    );
  }
  return { ...contribution, path: normalizedPath };
}

type CanvasExtensionModuleLoader = (
  source: string,
) => Promise<CanvasExtensionModule>;

interface CanvasExtensionsRuntimeProviderProps {
  children: React.ReactNode;
  /** Test seam for environments that cannot import browser Blob URLs. */
  moduleLoader?: CanvasExtensionModuleLoader;
}

export function CanvasExtensionsRuntimeProvider({
  children,
  moduleLoader = loadCanvasExtensionModule,
}: CanvasExtensionsRuntimeProviderProps) {
  const active = useActiveBackend();
  const navigate = useNavigate();
  const query = useCanvasExtensions();
  const [pages, setPages] = React.useState<RegisteredCanvasExtensionPage[]>([]);
  const [errors, setErrors] = React.useState<ReadonlyMap<string, string>>(
    new Map(),
  );
  const [activating, setActivating] = React.useState(false);

  const enabledExtensions = React.useMemo(
    () => (query.data ?? []).filter((extension) => extension.enabled),
    [query.data],
  );

  // `useActiveBackend` can synthesize a fresh `backend` object on every render
  // (e.g. when mounted without an <ActiveBackendProvider>), and refetches
  // produce new extension arrays with identical content. The activation effect
  // therefore keys on this value signature — backend identity plus the enabled
  // inventory — and reads the current objects from refs, so referential churn
  // never tears down and re-activates extensions.
  const activationSignature = React.useMemo(
    () =>
      JSON.stringify({
        backendId: active.backend.id,
        backendKind: active.backend.kind,
        connectionRevision: active.backend.connectionRevision ?? 0,
        orgId: active.orgId,
        extensions: enabledExtensions.map((extension) => ({
          name: extension.name,
          version: extension.version,
          resolvedRef: extension.resolved_ref ?? null,
          pages: extension.manifest?.contributes?.pages ?? [],
        })),
      }),
    [
      active.backend.id,
      active.backend.kind,
      active.backend.connectionRevision,
      active.orgId,
      enabledExtensions,
    ],
  );
  const activeRef = React.useRef(active);
  activeRef.current = active;
  const enabledExtensionsRef = React.useRef(enabledExtensions);
  enabledExtensionsRef.current = enabledExtensions;

  React.useEffect(() => {
    let cancelled = false;
    const disposers: CanvasExtensionDispose[] = [];
    const { backend, orgId } = activeRef.current;
    const extensionsToActivate = enabledExtensionsRef.current;
    setPages([]);
    setErrors(new Map());
    setActivating(extensionsToActivate.length > 0);

    const activateExtension = async (
      extension: InstalledCanvasExtensionInfo,
    ) => {
      const registeredPages = new Map<string, RegisteredCanvasExtensionPage>();
      const registrationDisposers: CanvasExtensionDispose[] = [];
      try {
        const source = await CanvasExtensionsService.fetchBundle(
          extension.name,
          backend,
        );
        if (cancelled) return;
        const extensionModule = await moduleLoader(source);
        if (cancelled) return;

        const host: CanvasExtensionHost = {
          apiVersion: CANVAS_EXTENSION_HOST_API_VERSION,
          extension: Object.freeze({
            name: extension.name,
            version: extension.version,
            resolvedRef: extension.resolved_ref ?? null,
          }),
          backend: Object.freeze({
            id: backend.id,
            kind: backend.kind,
            orgId,
          }),
          registerPage: (contributionId, mount) => {
            if (registeredPages.has(contributionId)) {
              throw new Error(
                `Extension ${extension.name} registered page "${contributionId}" more than once.`,
              );
            }
            const contribution = getDeclaredPage(extension, contributionId);
            const page: RegisteredCanvasExtensionPage = {
              extension,
              contribution,
              mount,
              href: buildCanvasExtensionPageHref(
                extension.name,
                contribution.path,
              ),
            };
            registeredPages.set(contributionId, page);
            const unregister = () => registeredPages.delete(contributionId);
            registrationDisposers.push(unregister);
            return unregister;
          },
          navigate: (path) => navigate(path),
          agentServer: {
            request: (request) =>
              CanvasExtensionsService.requestAgentServer(request, backend),
          },
        };

        const disposeActivation = await extensionModule.activate(host);
        if (cancelled) {
          if (typeof disposeActivation === "function") disposeActivation();
          return;
        }
        if (typeof disposeActivation === "function") {
          disposers.push(disposeActivation);
        }
        disposers.push(...registrationDisposers);
        setPages((current) => [
          ...current.filter((page) => page.extension.name !== extension.name),
          ...registeredPages.values(),
        ]);
      } catch (error) {
        registrationDisposers.forEach((dispose) => dispose());
        if (cancelled) return;
        const message =
          error instanceof Error
            ? error.message
            : "Extension activation failed.";
        setErrors((current) => {
          const next = new Map(current);
          next.set(extension.name, message);
          return next;
        });
      }
    };

    void Promise.all(extensionsToActivate.map(activateExtension)).finally(
      () => {
        if (!cancelled) setActivating(false);
      },
    );

    return () => {
      cancelled = true;
      setPages([]);
      for (const dispose of disposers.reverse()) {
        try {
          dispose();
        } catch (error) {
          console.error("Canvas Extension cleanup failed", error);
        }
      }
    };
  }, [activationSignature, moduleLoader, navigate]);

  const value = React.useMemo(
    () => ({ pages, activating, errors }),
    [pages, activating, errors],
  );

  return (
    <CanvasExtensionsRuntimeContext.Provider value={value}>
      {children}
    </CanvasExtensionsRuntimeContext.Provider>
  );
}
