/**
 * Reo.dev tracking service for SaaS mode
 * Tracks developer activity and engagement in the product
 * Using CDN approach for better TypeScript compatibility
 */

import EventLogger from "./event-logger";

export interface ReoIdentity {
  username: string;
  type: "github" | "email";
  other_identities?: Array<{
    username: string;
    type: "github" | "email";
  }>;
  firstname?: string;
  lastname?: string;
  company?: string;
}

const REO_CLIENT_ID = "6bac7145b4ee6ec";

class ReoService {
  private initialized = false;

  private scriptLoaded = false;

  /**
   * Load and initialize the Reo.dev tracking script from CDN
   */
  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      // Load the Reo script dynamically from CDN
      await this.loadScript();

      // Initialize Reo with client ID
      if (window.Reo) {
        window.Reo.init({ clientID: REO_CLIENT_ID });
        this.initialized = true;
      }
    } catch (error) {
      EventLogger.error(`Failed to initialize Reo.dev tracking: ${error}`);
    }
  }

  /**
   * Load the Reo.dev script from CDN
   */
  private loadScript(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.scriptLoaded) {
        resolve();
        return;
      }

      const script = document.createElement("script");
      script.src = `https://static.reo.dev/${REO_CLIENT_ID}/reo.js`;
      script.defer = true;

      script.onload = () => {
        this.scriptLoaded = true;
        resolve();
      };

      script.onerror = () => {
        reject(new Error("Failed to load Reo.dev script"));
      };

      document.head.appendChild(script);
    });
  }

  /**
   * Identify a user in Reo.dev tracking
   * Should be called after successful login
   */
  identify(identity: ReoIdentity): void {
    if (!this.initialized) {
      EventLogger.warning("Reo.dev not initialized. Call init() first.");
      return;
    }

    try {
      if (window.Reo) {
        window.Reo.identify(identity);
      }
    } catch (error) {
      EventLogger.error(`Failed to identify user in Reo.dev: ${error}`);
    }
  }

  /**
   * Check if Reo.dev is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }
}

const reoService = new ReoService();

export default reoService;
