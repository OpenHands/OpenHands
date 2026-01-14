interface Window {
  __APP_MODE__?: "saas" | "oss";
  __GITHUB_CLIENT_ID__?: string | null;
  Reo?: {
    init: (config: { clientID: string }) => void;
    identify: (identity: {
      username: string;
      type: "github" | "email";
      other_identities?: Array<{
        username: string;
        type: "github" | "email";
      }>;
      firstname?: string;
      lastname?: string;
      company?: string;
    }) => void;
  };
  grecaptcha?: {
    enterprise: {
      ready: (callback: () => void) => void;
      execute: (
        siteKey: string,
        options: { action: string },
      ) => Promise<string>;
    };
  };
}
