export function shouldStartMockWorker({
  mockApi = import.meta.env.VITE_MOCK_API,
  hasWindow = typeof window !== "undefined",
}: {
  mockApi?: string;
  hasWindow?: boolean;
} = {}): boolean {
  return hasWindow && mockApi === "true";
}
