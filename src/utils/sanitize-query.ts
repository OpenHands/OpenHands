export const sanitizeQuery = (query: string) =>
  query
    .trim()
    .replace(/https?:\/\//, "")
    .replace(/github.com\//, "")
    .replace(/\.git$/, "")
    .toLowerCase();
