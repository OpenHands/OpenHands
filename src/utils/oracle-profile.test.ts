import { describe, expect, it } from "vitest";
import { isOracleProfileName, ORACLE_PROFILE_NAME } from "./oracle-profile";

describe("Oracle profile name", () => {
  it("uses the SDK-reserved profile name", () => {
    expect(ORACLE_PROFILE_NAME).toBe("oracle");
    expect(isOracleProfileName("oracle")).toBe(true);
  });

  it("does not reserve other profile names", () => {
    expect(isOracleProfileName("oracle-copy")).toBe(false);
    expect(isOracleProfileName("Oracle")).toBe(false);
  });
});
