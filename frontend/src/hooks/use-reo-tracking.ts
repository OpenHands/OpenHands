import React from "react";
import { useConfig } from "./query/use-config";
import { useGitUser } from "./query/use-git-user";
import { getLoginMethod, LoginMethod } from "#/utils/local-storage";
import reoService, { ReoIdentity } from "#/utils/reo";
import { isProductionDomain } from "#/utils/utils";

/**
 * Maps login method to Reo identity type
 */
const mapLoginMethodToReoType = (method: LoginMethod): ReoIdentity["type"] => {
  // Reo is not supporting gitlab and bitbucket.
  switch (method) {
    case LoginMethod.GITHUB:
      return "github";
    case LoginMethod.ENTERPRISE_SSO:
      return "email";
    default:
      return "email";
  }
};

/**
 * Creates email identity object if email is available
 */
const buildEmailIdentity = (
  email?: string | null,
): ReoIdentity["other_identities"] => {
  if (!email) {
    return undefined;
  }

  return [
    {
      username: email,
      type: "email",
    },
  ];
};

/**
 * Parses full name into firstname and lastname
 * Handles cases where name might be empty or only have one part
 */
const parseNameFields = (
  fullName?: string | null,
): { firstname?: string; lastname?: string } => {
  if (!fullName) {
    return {};
  }

  const [firstname, ...rest] = fullName.split(" ");
  if (!firstname) {
    return {};
  }

  return {
    firstname,
    lastname: rest.length > 0 ? rest.join(" ") : undefined,
  };
};

/**
 * Builds complete Reo identity from user data and login method
 */
const buildReoIdentity = (
  user: {
    login: string;
    email?: string | null;
    name?: string | null;
    company?: string | null;
  },
  loginMethod: LoginMethod,
): ReoIdentity => {
  const { firstname, lastname } = parseNameFields(user.name);

  return {
    username: user.login,
    type: mapLoginMethodToReoType(loginMethod),
    other_identities: buildEmailIdentity(user.email),
    firstname,
    lastname,
    company: user.company || undefined,
  };
};

/**
 * Hook to handle Reo.dev tracking integration
 * Only active in SaaS mode
 */
export const useReoTracking = () => {
  const { data: config } = useConfig();
  const { data: user } = useGitUser();
  const [hasIdentified, setHasIdentified] = React.useState(false);

  // Initialize Reo.dev when in SaaS mode and on the correct domain
  React.useEffect(() => {
    const initReo = async () => {
      if (
        config?.APP_MODE === "saas" &&
        isProductionDomain() &&
        !reoService.isInitialized()
      ) {
        await reoService.init();
      }
    };

    initReo();
  }, [config?.APP_MODE]);

  // Identify user when user data is available and we're in SaaS mode on correct domain
  React.useEffect(() => {
    if (
      config?.APP_MODE !== "saas" ||
      !isProductionDomain() ||
      !user ||
      hasIdentified ||
      !reoService.isInitialized()
    ) {
      return;
    }

    const loginMethod = getLoginMethod();
    if (!loginMethod) {
      return;
    }

    // Build identity payload from user data
    const identity = buildReoIdentity(user, loginMethod);

    // Identify user in Reo
    reoService.identify(identity);
    setHasIdentified(true);
  }, [config?.APP_MODE, user, hasIdentified]);
};
