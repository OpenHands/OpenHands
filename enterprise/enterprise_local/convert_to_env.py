import base64
import os
import sys

import yaml


def convert_yaml_to_env(yaml_file, target_parameters, output_env_file, prefix):
    """Converts a YAML file into .env file format for specified target parameters under 'stringData' and 'data'.

    :param yaml_file: Path to the YAML file.
    :param target_parameters: List of keys to extract from the YAML file.
    :param output_env_file: Path to the output .env file.
    :param prefix: Prefix for environment variables.
    """
    try:
        # Load the YAML file
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)

        # Extract sections
        string_data = yaml_data.get('stringData', None)
        data = yaml_data.get('data', None)

        if string_data:
            env_source = string_data
            process_base64 = False
        elif data:
            env_source = data
            process_base64 = True
        else:
            print(
                "Error: Neither 'stringData' nor 'data' section found in the YAML file."
            )
            return

        env_lines = []

        for param in target_parameters:
            if param in env_source:
                value = env_source[param]
                if process_base64:
                    try:
                        decoded_value = base64.b64decode(value).decode('utf-8')
                        formatted_value = (
                            decoded_value.replace('\n', '\\n')
                            if '\n' in decoded_value
                            else decoded_value
                        )
                    except Exception as decode_error:
                        print(f"Error decoding base64 for '{param}': {decode_error}")
                        continue
                else:
                    formatted_value = (
                        value.replace('\n', '\\n')
                        if isinstance(value, str) and '\n' in value
                        else value
                    )

                new_key = prefix + param.upper().replace('-', '_')
                env_lines.append(f'{new_key}={formatted_value}')
            else:
                print(
                    f"Warning: Parameter '{param}' not found in the selected section."
                )

        # Write to the .env file
        with open(output_env_file, 'a') as env_file:
            env_file.write('\n'.join(env_lines) + '\n')

    except Exception as e:
        print(f'Error: {e}')


lite_llm_api_key = os.getenv('LITE_LLM_API_KEY')
if not lite_llm_api_key:
    print('Set the LITE_LLM_API_KEY environment variable to your API key')
    sys.exit(1)

yaml_file = 'github_decrypted.yaml'
target_parameters = ['client-id', 'client-secret', 'webhook-secret', 'private-key']
output_env_file = './enterprise/.env'

if os.path.exists(output_env_file):
    os.remove(output_env_file)
convert_yaml_to_env(yaml_file, target_parameters, output_env_file, 'GITHUB_APP_')
os.remove(yaml_file)

yaml_file = 'keycloak_realm_decrypted.yaml'
target_parameters = ['client-id', 'client-secret', 'provider-name', 'realm-name']
convert_yaml_to_env(yaml_file, target_parameters, output_env_file, 'KEYCLOAK_')
os.remove(yaml_file)

yaml_file = 'keycloak_admin_decrypted.yaml'
target_parameters = ['admin-password']
convert_yaml_to_env(yaml_file, target_parameters, output_env_file, 'KEYCLOAK_')
os.remove(yaml_file)

lines = []
lines.append('KEYCLOAK_SERVER_URL=https://auth.staging.all-hands.dev/')
lines.append('KEYCLOAK_SERVER_URL_EXT=https://auth.staging.all-hands.dev/')
lines.append('OPENHANDS_CONFIG_CLS=server.config.SaaSServerConfig')
lines.append(
    'OPENHANDS_GITHUB_SERVICE_CLS=integrations.github.github_service.SaaSGitHubService'
)
lines.append(
    'OPENHANDS_GITLAB_SERVICE_CLS=integrations.gitlab.gitlab_service.SaaSGitLabService'
)
lines.append(
    'OPENHANDS_BITBUCKET_SERVICE_CLS=integrations.bitbucket.bitbucket_service.SaaSBitBucketService'
)
lines.append(
    'OPENHANDS_CONVERSATION_VALIDATOR_CLS=storage.saas_conversation_validator.SaasConversationValidator'
)
lines.append('POSTHOG_CLIENT_KEY=test')
lines.append('ENABLE_PROACTIVE_CONVERSATION_STARTERS=true')
lines.append('MAX_CONCURRENT_CONVERSATIONS=10')
lines.append('LITE_LLM_API_URL=https://llm-proxy.eval.all-hands.dev')
lines.append('LITELLM_DEFAULT_MODEL=litellm_proxy/claude-opus-4-5-20251101')
lines.append(f'LITE_LLM_API_KEY={lite_llm_api_key}')
lines.append('LOCAL_DEPLOYMENT=true')
lines.append('DB_HOST=localhost')

with open(output_env_file, 'a') as env_file:
    env_file.write('\n'.join(lines))

print(f'.env file created at: {output_env_file}')
