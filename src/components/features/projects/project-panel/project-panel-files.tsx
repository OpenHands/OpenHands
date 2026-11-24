import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Paperclip, Upload } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import ProjectsService, {
  projectIdKey,
  type ProjectArtifact,
} from "#/api/projects-service/projects-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  projectsQueryRoot,
  useProjectArtifacts,
  useProjectConversations,
} from "#/hooks/query/use-projects";
import { useAttachProjectFile } from "#/hooks/mutation/use-attach-project-file";
import { ProjectFilePreview } from "#/components/features/projects/project-files";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { cn } from "#/utils/utils";
import { ProjectPanelGroup } from "./project-panel-group";

function FileRow({
  projectId,
  artifact,
  maker,
  isOpen,
  onToggle,
  onAttach,
  isAttaching,
}: {
  projectId: string;
  artifact: ProjectArtifact;
  maker: string;
  isOpen: boolean;
  onToggle: () => void;
  onAttach: () => void;
  isAttaching: boolean;
}) {
  const { t } = useTranslation("openhands");
  return (
    <li
      data-testid="project-panel-file"
      data-source={artifact.source}
      className={cn(
        "-mx-2 flex flex-col gap-2 rounded-lg px-2 py-1.5",
        isOpen ? "bg-surface" : "hover:bg-contrast/5",
      )}
    >
      <div className="flex items-start gap-2">
        <button
          type="button"
          aria-expanded={isOpen}
          className="flex min-w-0 flex-1 flex-col gap-0.5 text-left"
          onClick={onToggle}
        >
          <span className="truncate text-sm text-foreground">
            {artifact.name}
          </span>
          <span className="truncate text-xs text-muted">
            {maker} · {formatTimeDelta(artifact.updated_at)}
          </span>
        </button>
        <button
          type="button"
          data-testid="project-panel-file-attach"
          className="mt-0.5 inline-flex shrink-0 items-center gap-1 rounded-md border border-control px-1.5 py-0.5 text-xs text-muted hover:text-foreground disabled:opacity-50"
          aria-label={t(I18nKey.PROJECTS$ATTACH_NAMED, { name: artifact.name })}
          title={t(I18nKey.PROJECTS$ATTACH_NAMED, { name: artifact.name })}
          disabled={isAttaching}
          aria-busy={isAttaching}
          onClick={onAttach}
        >
          <Paperclip width={12} height={12} aria-hidden />
          {t(I18nKey.PROJECTS$ATTACH)}
        </button>
      </div>
      {isOpen ? (
        <ProjectFilePreview projectId={projectId} artifact={artifact} />
      ) : null}
    </li>
  );
}

/**
 * The project's files, newest first, split into the agents' and the user's.
 * A file previews in place and attaches to this conversation's message;
 * download and delete stay on the project page.
 */
export function ProjectPanelFiles({ projectId }: { projectId: string }) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const { data: artifacts, isError } = useProjectArtifacts(projectId);
  const { data: board } = useProjectConversations(projectId);
  const [open, setOpen] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const attach = useAttachProjectFile(projectId);

  const upload = useMutation({
    mutationFn: (file: File) => ProjectsService.uploadArtifact(projectId, file),
    meta: { disableToast: true },
    onSuccess: (artifact) => setOpen(artifact.name),
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) }),
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$UPLOAD_FAILED)),
  });

  const titles = React.useMemo(() => {
    const map = new Map<string, string>();
    if (board?.coordinator) {
      map.set(
        projectIdKey(board.coordinator.conversation_id),
        t(I18nKey.PROJECTS$COORDINATOR),
      );
    }
    for (const c of board?.conversations ?? []) {
      if (c.title) map.set(projectIdKey(c.conversation_id), c.title);
    }
    return map;
  }, [board, t]);

  const maker = (artifact: ProjectArtifact) => {
    if (!artifact.conversation_id) {
      return t(I18nKey.PROJECTS$FILE_FROM_A_CONVERSATION);
    }
    const id = projectIdKey(artifact.conversation_id);
    return titles.get(id) ?? id.slice(0, 8);
  };

  const sorted = [...(artifacts ?? [])].sort((a, b) =>
    b.updated_at.localeCompare(a.updated_at),
  );
  const fromAgents = sorted.filter((a) => a.source === "agent");
  const fromUser = sorted.filter((a) => a.source === "user");

  const rows = (items: ProjectArtifact[], makerOf: typeof maker) =>
    items.map((artifact) => (
      <FileRow
        key={artifact.name}
        projectId={projectId}
        artifact={artifact}
        maker={makerOf(artifact)}
        isOpen={open === artifact.name}
        onToggle={() =>
          setOpen((current) =>
            current === artifact.name ? null : artifact.name,
          )
        }
        onAttach={() => attach.mutate(artifact.name)}
        isAttaching={attach.isPending && attach.variables === artifact.name}
      />
    ));

  return (
    <div data-testid="project-panel-files" className="flex flex-col gap-3">
      <button
        type="button"
        data-testid="project-panel-file-upload"
        className="flex items-center gap-1.5 rounded-lg border border-dashed border-control px-3 py-1.5 text-sm text-muted hover:text-foreground disabled:opacity-50"
        disabled={upload.isPending}
        aria-busy={upload.isPending}
        onClick={() => inputRef.current?.click()}
      >
        <Upload width={14} height={14} aria-hidden />
        {t(I18nKey.PROJECTS$UPLOAD)}
      </button>
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        data-testid="project-panel-file-input"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) upload.mutate(file);
          // eslint-disable-next-line no-param-reassign
          event.target.value = "";
        }}
      />
      {isError ? (
        <p className="text-sm text-danger">
          {t(I18nKey.PROJECTS$FILES_LOAD_FAILED)}
        </p>
      ) : null}
      {!isError && artifacts && !sorted.length ? (
        <p className="text-sm text-muted">{t(I18nKey.PROJECTS$FILES_EMPTY)}</p>
      ) : null}
      <ProjectPanelGroup
        id="files-agents"
        label={t(I18nKey.PROJECTS$FILES_FROM_AGENTS)}
        count={fromAgents.length}
      >
        {rows(fromAgents, maker)}
      </ProjectPanelGroup>
      <ProjectPanelGroup
        id="files-user"
        label={t(I18nKey.PROJECTS$FILES_FROM_YOU)}
        count={fromUser.length}
      >
        {rows(fromUser, () => t(I18nKey.PROJECTS$FILE_FROM_YOU))}
      </ProjectPanelGroup>
    </div>
  );
}
