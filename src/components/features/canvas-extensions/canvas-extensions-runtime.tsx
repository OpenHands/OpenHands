import React from "react";
import { useNavigate } from "react-router";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { loadCanvasExtensionModule } from "#/extensions/canvas-extension-module-loader";
import { useCanvasExtensions } from "#/hooks/query/use-canvas-extensions";
import { subscribeConversationContextChangeRequested } from "#/services/conversation-context-events";
import {
  CANVAS_EXTENSION_HOST_API_VERSION,
  type CanvasExtensionCompanion,
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
  icon?: "cat";
}

export interface RegisteredCanvasExtensionCompanion extends CanvasExtensionCompanion {
  extensionName: string;
  scope: string;
  navigate: (path: string) => void;
}

interface CanvasExtensionsRuntimeValue {
  pages: RegisteredCanvasExtensionPage[];
  companions: RegisteredCanvasExtensionCompanion[];
  activating: boolean;
  errors: ReadonlyMap<string, string>;
}

const EMPTY_RUNTIME: CanvasExtensionsRuntimeValue = {
  pages: [],
  companions: [],
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
  const navigateRef = React.useRef(navigate);
  navigateRef.current = navigate;
  const query = useCanvasExtensions();
  const [pages, setPages] = React.useState<RegisteredCanvasExtensionPage[]>([]);
  const [companions, setCompanions] = React.useState<
    RegisteredCanvasExtensionCompanion[]
  >([]);
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
    setCompanions([]);
    setErrors(new Map());
    setActivating(extensionsToActivate.length > 0);

    const activateExtension = async (
      extension: InstalledCanvasExtensionInfo,
    ) => {
      const registeredPages = new Map<string, RegisteredCanvasExtensionPage>();
      const registeredCompanions = new Map<
        string,
        RegisteredCanvasExtensionCompanion
      >();
      const registrationDisposers: CanvasExtensionDispose[] = [];
      let activated = false;
      const navigateTo = (path: string) => {
        if (!cancelled) navigateRef.current(path);
      };
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
          registerPage: (contributionId, mount, options) => {
            if (cancelled) throw new Error("App activation has ended.");
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
              icon: options?.icon,
              href: buildCanvasExtensionPageHref(
                extension.name,
                contribution.path,
              ),
            };
            registeredPages.set(contributionId, page);
            if (activated) setPages((current) => [...current, page]);
            const unregister = () => {
              if (registeredPages.get(contributionId) !== page) return;
              registeredPages.delete(contributionId);
              if (activated && !cancelled) {
                setPages((current) => current.filter((item) => item !== page));
              }
            };
            registrationDisposers.push(unregister);
            return unregister;
          },
          registerCompanion: ({ id, mount }) => {
            if (cancelled) throw new Error("App activation has ended.");
            if (!isValidSegment(id) || registeredCompanions.has(id)) {
              throw new Error(
                `App ${extension.name} registered an invalid or duplicate companion "${id}".`,
              );
            }
            const companion = {
              id,
              mount,
              extensionName: extension.name,
              scope: activationSignature,
              navigate: navigateTo,
            };
            registeredCompanions.set(id, companion);
            if (activated) setCompanions((current) => [...current, companion]);
            const unregister = () => {
              if (registeredCompanions.get(id) !== companion) return;
              registeredCompanions.delete(id);
              if (activated && !cancelled) {
                setCompanions((current) =>
                  current.filter((item) => item !== companion),
                );
              }
            };
            registrationDisposers.push(unregister);
            return unregister;
          },
          onConversationContextChangeRequested: (listener) => {
            if (cancelled) throw new Error("App activation has ended.");
            const unsubscribe = subscribeConversationContextChangeRequested(
              {
                backendId: backend.id,
                orgId,
                connectionRevision: backend.connectionRevision ?? 0,
              },
              (event) => {
                if (!cancelled) listener(event);
              },
            );
            registrationDisposers.push(unsubscribe);
            return unsubscribe;
          },
          navigate: navigateTo,
          agentServer: {
            request: (request) => {
              // Resource cleanup may settle after disposal. Always retain its
              // owning backend rather than sending it to the newly active one.
              return CanvasExtensionsService.requestAgentServer(
                request,
                backend,
              );
            },
          },
        };

        const disposeActivation = await extensionModule.activate(host);
        if (cancelled) {
          if (typeof disposeActivation === "function") disposeActivation();
          registrationDisposers.forEach((dispose) => dispose());
          return;
        }
        if (typeof disposeActivation === "function") {
          disposers.push(disposeActivation);
        }
        disposers.push(() => {
          registrationDisposers.forEach((dispose) => dispose());
        });
        activated = true;
        setPages((current) => [
          ...current.filter((page) => page.extension.name !== extension.name),
          ...registeredPages.values(),
        ]);
        setCompanions((current) => [
          ...current.filter((item) => item.extensionName !== extension.name),
          ...registeredCompanions.values(),
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
      setCompanions([]);
      for (const dispose of disposers.reverse()) {
        try {
          dispose();
        } catch (error) {
          console.error("Canvas Extension cleanup failed", error);
        }
      }
    };
  }, [activationSignature, moduleLoader]);

  const scopedCompanions = React.useMemo(
    () => companions.filter((item) => item.scope === activationSignature),
    [companions, activationSignature],
  );
  const value = React.useMemo(
    () => ({ pages, companions: scopedCompanions, activating, errors }),
    [pages, scopedCompanions, activating, errors],
  );

  return (
    <CanvasExtensionsRuntimeContext.Provider value={value}>
      {children}
    </CanvasExtensionsRuntimeContext.Provider>
  );
}
