import { describe, expect, it } from "vitest";
import {
  AGENT_SERVER_UI_SCOPE_SELECTOR,
  transformAgentServerUISelector,
} from "#/styles/agent-server-ui-style-scope";

describe("transformAgentServerUISelector", () => {
  it("prefixes ordinary selectors under the scoped root", () => {
    expect(
      transformAgentServerUISelector(
        AGENT_SERVER_UI_SCOPE_SELECTOR,
        ".button-base",
        `${AGENT_SERVER_UI_SCOPE_SELECTOR} .button-base`,
      ),
    ).toBe(`${AGENT_SERVER_UI_SCOPE_SELECTOR} .button-base`);
  });

  it("replaces :host selectors with the scoped root", () => {
    expect(
      transformAgentServerUISelector(
        AGENT_SERVER_UI_SCOPE_SELECTOR,
        ":host",
        `${AGENT_SERVER_UI_SCOPE_SELECTOR} :host`,
      ),
    ).toBe(AGENT_SERVER_UI_SCOPE_SELECTOR);
  });

  it.each([":root", "body", "html"])(
    "maps %s selectors directly to the scoped root",
    (selector) => {
      expect(
        transformAgentServerUISelector(
          AGENT_SERVER_UI_SCOPE_SELECTOR,
          selector,
          `${AGENT_SERVER_UI_SCOPE_SELECTOR} ${selector}`,
        ),
      ).toBe(AGENT_SERVER_UI_SCOPE_SELECTOR);
    },
  );

  it("does not double-prefix selectors that are already scoped", () => {
    const selector = `${AGENT_SERVER_UI_SCOPE_SELECTOR} .xterm`;

    expect(
      transformAgentServerUISelector(
        AGENT_SERVER_UI_SCOPE_SELECTOR,
        selector,
        `${AGENT_SERVER_UI_SCOPE_SELECTOR} ${selector}`,
      ),
    ).toBe(selector);
  });
});
