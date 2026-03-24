import toast from "react-hot-toast";
import i18next from "i18next";
import { UseMutationResult } from "@tanstack/react-query";
import {
  TOAST_OPTIONS,
  displaySuccessToast,
  displayErrorToast,
} from "./custom-toast-handlers";

export type ToastMessages<TData> = {
  success?: string | ((data: TData) => string) | false;
  error?: string | ((error: Error) => string) | false;
  loading?: string;
};

export async function mutateWithToast<TData, TVariables>(
  mutation: UseMutationResult<TData, Error, TVariables>,
  variables: TVariables,
  messages: ToastMessages<TData>,
): Promise<TData> {
  const { success, error, loading } = messages;

  let loadingToastId: string | undefined;
  if (loading) {
    loadingToastId = toast.loading(loading, TOAST_OPTIONS);
  }

  try {
    const result = await mutation.mutateAsync(variables);

    if (loadingToastId) toast.dismiss(loadingToastId);

    if (success !== false) {
      const message = typeof success === "function" ? success(result) : success;
      if (message) displaySuccessToast(message);
    }

    return result;
  } catch (err) {
    if (loadingToastId) toast.dismiss(loadingToastId);

    if (error !== false) {
      let message: string;
      if (typeof error === "function") {
        message = error(err as Error);
      } else if (error) {
        message = error;
      } else if (err instanceof Error) {
        message = err.message;
      } else {
        message = i18next.t("ERROR$GENERIC");
      }
      displayErrorToast(message);
    }

    throw err;
  }
}
