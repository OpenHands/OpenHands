import { describe, expect, it } from "vitest";
import { Provider } from "#/types/settings";
import { convertRawProvidersToList } from "#/utils/convert-raw-providers-to-list";

describe("convertRawProvidersToList", () => {
  it("should convert raw provider tokens to a list of providers", () => {
    const example1: Partial<Record<Provider, string | null>> | undefined = {
      github: "test-token",
      gitlab: "test-token",
      azure_devops: "test-token",
    };
    const example2: Partial<Record<Provider, string | null>> | undefined = {
      github: "",
    };
    const example3: Partial<Record<Provider, string | null>> | undefined = {
      gitlab: null,
    };
    const example4: Partial<Record<Provider, string | null>> | undefined = {
      azure_devops: "test-token",
    };

    expect(convertRawProvidersToList(example1)).toEqual(["github", "gitlab", "azure_devops"]);
    expect(convertRawProvidersToList(example2)).toEqual(["github"]);
    expect(convertRawProvidersToList(example3)).toEqual(["gitlab"]);
    expect(convertRawProvidersToList(example4)).toEqual(["azure_devops"]);
  });
});
