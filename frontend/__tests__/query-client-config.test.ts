import { afterEach, describe, expect, it, vi } from "vitest";
import { AxiosError, AxiosHeaders } from "axios";
import { queryClient } from "#/query-client-config";

const make429 = () =>
  new AxiosError("Too Many Requests", "429", undefined, undefined, {
    status: 429,
    statusText: "Too Many Requests",
    headers: { "retry-after": "0" },
    config: { headers: new AxiosHeaders() },
    data: { detail: "rate_limited" },
  });

describe("queryClient mutation defaults", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    queryClient.clear();
  });

  it("should retry a mutation rejected with 429 (request was never processed)", async () => {
    const mutationFn = vi
      .fn()
      .mockRejectedValueOnce(make429())
      .mockResolvedValueOnce("ok");

    const result = await queryClient
      .getMutationCache()
      .build(queryClient, { mutationFn, meta: { disableToast: true } })
      .execute(undefined);

    expect(result).toBe("ok");
    expect(mutationFn).toHaveBeenCalledTimes(2);
  });

  it("should not retry a mutation rejected with a non-429 error", async () => {
    const error = new AxiosError("Server Error", "500", undefined, undefined, {
      status: 500,
      statusText: "",
      headers: {},
      config: { headers: new AxiosHeaders() },
      data: {},
    });
    const mutationFn = vi.fn().mockRejectedValue(error);

    await expect(
      queryClient
        .getMutationCache()
        .build(queryClient, { mutationFn, meta: { disableToast: true } })
        .execute(undefined),
    ).rejects.toBe(error);

    expect(mutationFn).toHaveBeenCalledTimes(1);
  });
});
