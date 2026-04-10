import { openHands } from "./open-hands-axios";

export interface SSHKey {
  id: string;
  key: string;
  label: string | null;
}

export interface SSHKeysListResponse {
  ssh_keys: SSHKey[];
}

export interface SSHKeyRequest {
  key: string;
  label?: string | null;
}

class SSHKeysService {
  static async listSSHKeys(): Promise<SSHKey[]> {
    const response =
      await openHands.get<SSHKeysListResponse>("/api/v1/ssh-keys");
    return response.data.ssh_keys;
  }

  static async addSSHKey(request: SSHKeyRequest): Promise<SSHKey> {
    const response = await openHands.post<SSHKey>("/api/v1/ssh-keys", request);
    return response.data;
  }

  static async updateSSHKey(
    keyId: string,
    request: SSHKeyRequest,
  ): Promise<SSHKey> {
    const response = await openHands.put<SSHKey>(
      `/api/v1/ssh-keys/${keyId}`,
      request,
    );
    return response.data;
  }

  static async deleteSSHKey(keyId: string): Promise<void> {
    await openHands.delete(`/api/v1/ssh-keys/${keyId}`);
  }
}

export default SSHKeysService;
