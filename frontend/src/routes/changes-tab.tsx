import { useTranslation } from "react-i18next";
import React from "react";
import { FileDiffViewer } from "#/components/features/diff-viewer/file-diff-viewer";
import { EmptyChangesMessage } from "#/components/features/diff-viewer/empty-changes-message";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import { I18nKey } from "#/i18n/declaration";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { RandomTip } from "#/components/features/tips/random-tip";
import { useAgentState } from "#/hooks/use-agent-state";

// Error message patterns
const GIT_REPO_ERROR_PATTERN = /not a git repository/i;

function StatusMessage({ children }: React.PropsWithChildren) {
  return (
    <div className="w-full h-full flex flex-col items-center text-center justify-center text-2xl text-tertiary-light">
      {children}
    </div>
  );
}

function GitChanges() {
  const { t } = useTranslation();
  const {
    data: gitChanges,
    isSuccess,
    isError,
    error,
    isLoading: loadingGitChanges,
  } = useUnifiedGetGitChanges();

  const [statusMessage, setStatusMessage] = React.useState<string[] | null>(
    null,
  );

  const { curAgentState } = useAgentState();
  const runtimeIsActive = !RUNTIME_INACTIVE_STATES.includes(curAgentState);

  const isNotGitRepoError =
    error && GIT_REPO_ERROR_PATTERN.test(retrieveAxiosErrorMessage(error));

  React.useEffect(() => {
    if (!runtimeIsActive) {
      setStatusMessage([I18nKey.DIFF_VIEWER$WAITING_FOR_RUNTIME]);
    } else if (error) {
      const errorMessage = retrieveAxiosErrorMessage(error);
      if (GIT_REPO_ERROR_PATTERN.test(errorMessage)) {
        setStatusMessage([
          I18nKey.DIFF_VIEWER$NOT_A_GIT_REPO,
          I18nKey.DIFF_VIEWER$ASK_OH,
        ]);
      } else {
        setStatusMessage([errorMessage]);
      }
    } else if (loadingGitChanges) {
      setStatusMessage([I18nKey.DIFF_VIEWER$LOADING]);
    } else {
      setStatusMessage(null);
    }
  }, [
    runtimeIsActive,
    isNotGitRepoError,
    loadingGitChanges,
    error,
    setStatusMessage,
  ]);

  return (
    <main className="h-full overflow-y-scroll p-4 md:pr-1.5 gap-3 flex flex-col items-center custom-scrollbar-always">
      {!isSuccess || !gitChanges.length ? (
        <div className="relative flex h-full w-full items-center">
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2">
            {statusMessage && (
              <StatusMessage>
                {statusMessage.map((msg) => (
                  <span key={msg}>{t(msg)}</span>
                ))}
              </StatusMessage>
            )}
            {!statusMessage && isSuccess && gitChanges.length === 0 && (
              <EmptyChangesMessage />
            )}
          </div>

          <div className="absolute inset-x-0 bottom-0">
            {!isError && gitChanges?.length === 0 && (
              <div className="max-w-2xl mb-4 text-m bg-tertiary rounded-xl p-4 text-left mx-auto">
                <RandomTip />
              </div>
            )}
          </div>
        </div>
      ) : (
        gitChanges
          .slice(0, 100)
          .map((change) => (
            <FileDiffViewer
              key={change.path}
              path={change.path}
              type={change.status}
            />
          ))
      )}
    </main>
  );
}

export default GitChanges;
