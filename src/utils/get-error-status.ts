import axios from "axios";

export const getErrorStatus = (error: unknown): number | undefined => {
  if (typeof error === "object" && error !== null && "status" in error) {
    const { status } = error as { status?: unknown };
    if (typeof status === "number") {
      return status;
    }
  }

  if (axios.isAxiosError(error)) {
    return error.response?.status;
  }

  return undefined;
};
