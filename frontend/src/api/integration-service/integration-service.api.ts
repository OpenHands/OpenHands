import { openHands } from "../open-hands-axios";
import {
  BitbucketDCResourcesResponse,
  BitbucketDCWebhookEnrollmentResult,
  BitbucketDCWebhookIdUpdateResult,
  BitbucketDCWebhookInstallationResult,
  BitbucketDCResourceIdentifier,
  GitLabResourcesResponse,
  UpdateBitbucketDCWebhookIdRequest,
  ReinstallWebhookRequest,
  ResourceIdentifier,
  ResourceInstallationResult,
} from "./integration-service.types";

export const integrationService = {
  /**
   * Get all GitLab projects and groups where the user has admin access
   * @returns Promise with list of resources and their webhook status
   */
  getGitLabResources: async (): Promise<GitLabResourcesResponse> => {
    const { data } = await openHands.get<GitLabResourcesResponse>(
      "/integration/gitlab/resources",
    );
    return data;
  },

  /**
   * Reinstall webhook on a specific GitLab resource
   * @param resource - Resource to reinstall webhook on
   * @returns Promise with installation result
   */
  reinstallGitLabWebhook: async ({
    resource,
  }: {
    resource: ResourceIdentifier;
  }): Promise<ResourceInstallationResult> => {
    const requestBody: ReinstallWebhookRequest = { resource };
    const { data } = await openHands.post<ResourceInstallationResult>(
      "/integration/gitlab/reinstall-webhook",
      requestBody,
    );
    return data;
  },

  /**
   * Get all Bitbucket Data Center repositories visible to the user with webhook enrollment status
   * @returns Promise with list of repositories and their webhook status
   */
  getBitbucketDCResources: async (): Promise<BitbucketDCResourcesResponse> => {
    const { data } = await openHands.get<BitbucketDCResourcesResponse>(
      "/integration/bitbucket-dc/resources",
    );
    return data;
  },

  /**
   * Enroll a Bitbucket Data Center repository webhook in OpenHands.
   * The returned secret and URL must be copied into Bitbucket Data Center manually.
   */
  enrollBitbucketDCWebhook: async ({
    resource,
  }: {
    resource: BitbucketDCResourceIdentifier;
  }): Promise<BitbucketDCWebhookEnrollmentResult> => {
    const { data } = await openHands.post<BitbucketDCWebhookEnrollmentResult>(
      "/integration/bitbucket-dc/enroll-webhook",
      { resource },
    );
    return data;
  },

  /**
   * Record the numeric Bitbucket Data Center webhook id after manual creation.
   * @deprecated Manual flow superseded by reinstallBitbucketDCWebhook, which
   *   auto-creates the webhook on BBDC and records the assigned id in one step.
   *   Kept for backward compatibility with older clients.
   */
  updateBitbucketDCWebhookId: async ({
    resource,
    webhookId,
  }: {
    resource: BitbucketDCResourceIdentifier;
    webhookId: string;
  }): Promise<BitbucketDCWebhookIdUpdateResult> => {
    const requestBody: UpdateBitbucketDCWebhookIdRequest = {
      resource,
      webhook_id: webhookId,
    };
    const { data } = await openHands.patch<BitbucketDCWebhookIdUpdateResult>(
      "/integration/bitbucket-dc/webhook-id",
      requestBody,
    );
    return data;
  },

  /**
   * Install or reinstall the webhook on a Bitbucket Data Center repository
   * via BBDC's REST API. Rotates the shared secret, idempotently creates or
   * updates the webhook on BBDC, and persists the result. Requires the
   * caller's BBDC OAuth token to have REPO_ADMIN scope.
   */
  reinstallBitbucketDCWebhook: async ({
    resource,
  }: {
    resource: BitbucketDCResourceIdentifier;
  }): Promise<BitbucketDCWebhookInstallationResult> => {
    const { data } = await openHands.post<BitbucketDCWebhookInstallationResult>(
      "/integration/bitbucket-dc/reinstall-webhook",
      { resource },
    );
    return data;
  },

  /**
   * Delete the webhook on Bitbucket Data Center and clear the local enrollment.
   */
  uninstallBitbucketDCWebhook: async ({
    resource,
  }: {
    resource: BitbucketDCResourceIdentifier;
  }): Promise<BitbucketDCWebhookInstallationResult> => {
    const { data } = await openHands.post<BitbucketDCWebhookInstallationResult>(
      "/integration/bitbucket-dc/uninstall-webhook",
      { resource },
    );
    return data;
  },
};
