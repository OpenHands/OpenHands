import { openHands } from "../open-hands-axios";
import { Settings, SettingsSchema } from "#/types/settings";

/**
 * Settings service for managing application settings
 */
class SettingsService {
  /**
   * Get the settings from the server or use the default settings if not found
   */
  static async getSettings(): Promise<Settings> {
    const { data } = await openHands.get<Settings>("/api/v1/settings");
    return data;
  }

  /**
   * Get the AgentSettings schema used to render schema-driven settings pages.
   */
  static async getSettingsSchema(): Promise<SettingsSchema> {
    const { data } = await openHands.get<SettingsSchema>(
      "/api/v1/settings/agent-schema",
    );
    return data;
  }

  static async getConversationSettingsSchema(): Promise<SettingsSchema> {
    const { data } = await openHands.get<SettingsSchema>(
      "/api/v1/settings/conversation-schema",
    );
    return data;
  }

  /**
   * Save the settings to the server. Only valid settings are saved.
   * @param settings - the settings to save
   */
  static async saveSettings(
    settings: Partial<Settings> & Record<string, unknown>,
  ): Promise<boolean> {
    const data = await openHands.post("/api/v1/settings", settings);
    return data.status === 200;
  }
}

export default SettingsService;
