import { useRef, useState } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { AgentProfileActionsMenu } from "#/components/features/settings/agent-profiles/agent-profile-actions-menu";
import { DEFAULT_ENABLED_SKILL_NAMES } from "@openhands/extensions/skills";
import {
  type AgentProfile,
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import {
  fromBehaviorFormValues,
  toBehaviorFormValues,
} from "#/components/features/settings/agent-profiles/profile-behavior-form";
import {
  type SettingsFieldSchema,
  type SettingsSchema,
} from "#/types/settings";

describe("agent-profile-actions-menu", () => {
  const defaultProps = {
    onEdit: vi.fn(),
    onSetActive: vi.fn(),
    onDelete: vi.fn(),
    isActive: false,
    isActivating: false,
    onClose: vi.fn(),
  };

  describe("AgentProfileActionsMenu", () => {
    it("renders Edit, Set as active, and Delete", () => {
      render(<AgentProfileActionsMenu {...defaultProps} />);

      const items = screen.getAllByRole("menuitem");
      expect(items.map((item) => item.getAttribute("data-testid"))).toEqual([
        "agent-profile-edit",
        "agent-profile-set-active",
        "agent-profile-delete",
      ]);
    });

    it("disables Set as active when already active", () => {
      render(<AgentProfileActionsMenu {...defaultProps} isActive />);

      expect(screen.getByTestId("agent-profile-set-active")).toBeDisabled();
    });

    describe("keyboard navigation when anchored to a row trigger", () => {
      // Mirrors AgentProfileRow: the trigger toggles the menu, which is portaled
      // to <body> against the trigger and unmounted on close.
      function AnchoredMenu({
        isActive = false,
        isActivating = false,
        onEdit = defaultProps.onEdit,
      }: {
        isActive?: boolean;
        isActivating?: boolean;
        onEdit?: () => void;
      }) {
        const triggerRef = useRef<HTMLButtonElement>(null);
        const [open, setOpen] = useState(false);
        return (
          <>
            <button
              ref={triggerRef}
              type="button"
              data-testid="agent-profile-menu-trigger"
              onClick={() => setOpen((value) => !value)}
            />
            {open && (
              <AgentProfileActionsMenu
                {...defaultProps}
                onEdit={onEdit}
                isActive={isActive}
                isActivating={isActivating}
                anchorRef={triggerRef}
                onClose={() => setOpen(false)}
              />
            )}
            <button type="button" data-testid="next-page-control" />
          </>
        );
      }

      it("moves focus to Edit when opened with the mouse", async () => {
        const user = userEvent.setup();
        render(<AnchoredMenu />);

        await user.click(screen.getByTestId("agent-profile-menu-trigger"));

        expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();
      });

      it("moves focus to Edit when opened with Enter on the trigger", async () => {
        const user = userEvent.setup();
        render(<AnchoredMenu />);

        screen.getByTestId("agent-profile-menu-trigger").focus();
        await user.keyboard("{Enter}");

        expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();
      });

      it("arrows across the disabled Set as active item on the active row", async () => {
        const user = userEvent.setup();
        render(<AnchoredMenu isActive />);

        await user.click(screen.getByTestId("agent-profile-menu-trigger"));
        expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

        await user.keyboard("{ArrowDown}");
        expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();

        await user.keyboard("{ArrowDown}");
        expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

        await user.keyboard("{ArrowUp}");
        expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();
      });

      it("cycles through all three items with wrap on a non-active row", async () => {
        const user = userEvent.setup();
        render(<AnchoredMenu />);

        await user.click(screen.getByTestId("agent-profile-menu-trigger"));
        expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

        await user.keyboard("{ArrowDown}");
        expect(screen.getByTestId("agent-profile-set-active")).toHaveFocus();

        await user.keyboard("{ArrowDown}");
        expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();

        await user.keyboard("{ArrowDown}");
        expect(screen.getByTestId("agent-profile-edit")).toHaveFocus();

        await user.keyboard("{ArrowUp}");
        expect(screen.getByTestId("agent-profile-delete")).toHaveFocus();
      });

      it("closes on Escape and returns focus to the trigger", async () => {
        const user = userEvent.setup();
        render(<AnchoredMenu />);
        await user.click(screen.getByTestId("agent-profile-menu-trigger"));

        await user.keyboard("{Escape}");

        expect(
          screen.queryByTestId("agent-profile-actions-menu"),
        ).not.toBeInTheDocument();
        expect(screen.getByTestId("agent-profile-menu-trigger")).toHaveFocus();
      });

      it("closes on Tab and moves focus to the control after the trigger", async () => {
        const user = userEvent.setup();
        render(<AnchoredMenu />);
        await user.click(screen.getByTestId("agent-profile-menu-trigger"));

        await user.keyboard("{Tab}");

        expect(
          screen.queryByTestId("agent-profile-actions-menu"),
        ).not.toBeInTheDocument();
        expect(screen.getByTestId("next-page-control")).toHaveFocus();
      });

      it("runs the focused action when Enter is pressed", async () => {
        const onEdit = vi.fn();
        const user = userEvent.setup();
        render(<AnchoredMenu onEdit={onEdit} />);
        await user.click(screen.getByTestId("agent-profile-menu-trigger"));

        await user.keyboard("{Enter}");

        expect(onEdit).toHaveBeenCalledTimes(1);
      });
    });
  });
});

describe("merge-agent-profile-save-input", () => {
  // Mirrors the seeded `default` profile: fields the minimal editor does NOT
  // model (condenser, verification, suffix, mcp_server_refs, the disabled_skills
  // deny-list) carry non-default values so a wipe-to-defaults would be visible.
  // `disabled_skills` rides untyped (the pinned ts-client predates it) — the
  // merge must still round-trip it, so the fixture includes it via the cast.
  const storedOpenHands = {
    schema_version: 1,
    id: "11111111-1111-1111-1111-111111111111",
    name: "default",
    revision: 3,
    agent_kind: "openhands",
    llm_profile_ref: "old-llm",
    agent: "CodeActAgent",
    system_message_suffix: "Be terse.",
    condenser: { kind: "NoOpCondenserSettings" },
    verification: {
      critic_enabled: true,
      critic_mode: "finish_and_message",
      enable_iterative_refinement: true,
      critic_threshold: 0.8,
      max_refinement_iterations: 5,
      critic_server_url: null,
      critic_model_name: null,
    },
    tools: [{ name: "terminal", params: {} }],
    tool_concurrency_limit: 4,
    mcp_server_refs: ["github"],
    disabled_skills: ["deploy-checklist"],
  } as unknown as AgentProfile;

  const storedAcp = {
    schema_version: 1,
    id: "22222222-2222-2222-2222-222222222222",
    name: "claude",
    revision: 1,
    agent_kind: "acp",
    acp_server: "claude-code",
    acp_model: "claude-opus-4-8",
    acp_session_mode: "bypassPermissions",
    acp_prompt_timeout: 900,
    acp_command: null,
    acp_args: null,
    mcp_server_refs: ["linear"],
  } as unknown as AgentProfile;

  describe("mergeAgentProfileSaveInput", () => {
    it("preserves unmodeled OpenHands fields under the edited ones", () => {
      const edited = {
        agent_kind: "openhands",
        tools: [{ name: "grep", params: {} }],
        llm_profile_ref: "new-llm",
      } as unknown as AgentProfileSaveInput;

      const merged = mergeAgentProfileSaveInput(storedOpenHands, edited);

      // Edited fields win.
      expect(merged).toMatchObject({
        agent_kind: "openhands",
        tools: [{ name: "grep", params: {} }],
        llm_profile_ref: "new-llm",
      });
      // Fields the editor doesn't model survive the whole-profile overwrite.
      expect(merged).toMatchObject({
        condenser: { kind: "NoOpCondenserSettings" },
        verification: { critic_enabled: true, critic_threshold: 0.8 },
        system_message_suffix: "Be terse.",
        mcp_server_refs: ["github"],
        disabled_skills: ["deploy-checklist"],
        tool_concurrency_limit: 4,
      });
    });

    it("preserves unmodeled ACP fields under the edited ones", () => {
      const edited: AgentProfileSaveInput = {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_model: "claude-sonnet-5",
        acp_command: null,
        acp_args: null,
      };

      const merged = mergeAgentProfileSaveInput(storedAcp, edited);

      expect(merged).toMatchObject({
        agent_kind: "acp",
        acp_model: "claude-sonnet-5",
        acp_session_mode: "bypassPermissions",
        acp_prompt_timeout: 900,
        mcp_server_refs: ["linear"],
      });
    });

    it("strips server-managed identity from the merge", () => {
      const merged = mergeAgentProfileSaveInput(storedOpenHands, {
        agent_kind: "openhands",
        llm_profile_ref: "new-llm",
      });

      // The path name is authoritative and the server preserves the namesake's
      // id / bumps revision itself; posting stale identity would only mislead.
      expect(merged).not.toHaveProperty("id");
      expect(merged).not.toHaveProperty("name");
      expect(merged).not.toHaveProperty("revision");
    });

    it("sends a clean variant payload on an agent_kind switch", () => {
      const edited: AgentProfileSaveInput = {
        agent_kind: "acp",
        acp_server: "codex",
        acp_model: null,
        acp_command: null,
        acp_args: null,
      };

      // openhands → acp: carrying stored openhands fields over would produce a
      // mongrel payload the server's extra="forbid" union rejects.
      expect(mergeAgentProfileSaveInput(storedOpenHands, edited)).toEqual(
        edited,
      );
    });

    it("starts a new OpenHands profile on the recommended catalog skills", () => {
      const edited: AgentProfileSaveInput = {
        agent_kind: "openhands",
        llm_profile_ref: "default",
      };

      expect(mergeAgentProfileSaveInput(null, edited)).toEqual({
        ...edited,
        public_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
      });
    });

    it("gives a profile switched to OpenHands the recommended catalog skills", () => {
      const storedAcp = {
        id: "p-2",
        name: "agent",
        revision: 1,
        agent_kind: "acp",
        acp_server: "codex",
      } as unknown as AgentProfile;

      expect(
        mergeAgentProfileSaveInput(storedAcp, {
          agent_kind: "openhands",
          llm_profile_ref: "default",
        }),
      ).toMatchObject({ public_skills: [...DEFAULT_ENABLED_SKILL_NAMES] });
    });

    it("keeps a stored or explicit skill selection", () => {
      const stored = {
        ...storedOpenHands,
        public_skills: ["github"],
      } as unknown as AgentProfile;

      expect(
        mergeAgentProfileSaveInput(stored, {
          agent_kind: "openhands",
          llm_profile_ref: "x",
        }),
      ).toMatchObject({ public_skills: ["github"] });
      expect(
        mergeAgentProfileSaveInput(null, {
          agent_kind: "openhands",
          llm_profile_ref: "x",
          public_skills: [],
        } as AgentProfileSaveInput),
      ).toMatchObject({ public_skills: [] });
    });
  });

  describe("mergeAgentProfileSaveInput tools", () => {
    it("round-trips stored tools the draft leaves out", () => {
      const stored = {
        ...storedOpenHands,
        tools: [{ name: "switch_llm", params: { x: 1 } }],
      } as unknown as AgentProfile;
      const edited = {
        agent_kind: "openhands",
        mcp_server_refs: null,
        tool_concurrency_limit: 2,
      } as AgentProfileSaveInput;

      expect(mergeAgentProfileSaveInput(stored, edited)).toMatchObject({
        tools: [{ name: "switch_llm", params: { x: 1 } }],
        tool_concurrency_limit: 2,
      });
    });

    it.each([[[{ name: "terminal", params: {} }]], [null]])(
      "replaces the stored tools with the draft's %j",
      (tools) => {
        const stored = {
          ...storedOpenHands,
          tools: [
            { name: "terminal", params: {} },
            { name: "switch_llm", params: {} },
          ],
        } as unknown as AgentProfile;
        const edited = {
          agent_kind: "openhands",
          mcp_server_refs: null,
          tools,
        } as unknown as AgentProfileSaveInput;

        expect(mergeAgentProfileSaveInput(stored, edited)).toMatchObject({
          tools,
          system_message_suffix: "Be terse.",
        });
      },
    );
  });
});

describe("profile-behavior-form", () => {
  const field = (
    key: string,
    value_type: SettingsFieldSchema["value_type"],
    defaultValue: SettingsFieldSchema["default"],
  ): SettingsFieldSchema =>
    ({
      key,
      label: key,
      section: key.split(".")[0],
      section_label: key.split(".")[0],
      value_type,
      default: defaultValue,
      choices: [],
      depends_on: [],
      prominence: "critical",
      secret: false,
      required: false,
    }) as SettingsFieldSchema;

  const SCHEMA: SettingsSchema = {
    model_name: "AgentSettings",
    sections: [
      {
        key: "verification",
        label: "Verification",
        fields: [
          field("verification.critic_enabled", "boolean", false),
          field("verification.critic_threshold", "number", 0.6),
          field("verification.critic_api_key", "string", null),
        ],
      },
    ],
  };

  describe("profile behavior form", () => {
    it("round-trips a stored section, filling a missing field with its default", () => {
      const values = toBehaviorFormValues(SCHEMA, "verification", {
        critic_enabled: true,
      });

      expect(fromBehaviorFormValues(SCHEMA, "verification", values)).toEqual({
        critic_enabled: true,
        critic_threshold: 0.6,
      });
    });

    it("never reads or writes the shared critic API key", () => {
      const values = toBehaviorFormValues(SCHEMA, "verification", {
        critic_api_key: "sk-stored",
      });

      expect(values).not.toHaveProperty("verification.critic_api_key");
      expect(
        fromBehaviorFormValues(SCHEMA, "verification", {
          ...values,
          "verification.critic_api_key": "sk-typed",
        }),
      ).not.toHaveProperty("critic_api_key");
    });
  });
});
