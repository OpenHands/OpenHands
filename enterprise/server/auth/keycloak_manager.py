from keycloak.keycloak_admin import KeycloakAdmin
from keycloak.keycloak_openid import KeycloakOpenID
from server.auth.constants import (
    KEYCLOAK_ADMIN_PASSWORD,
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_CLIENT_SECRET,
    KEYCLOAK_REALM_NAME,
    KEYCLOAK_SERVER_URL,
    KEYCLOAK_SERVER_URL_EXT,
)
from server.logger import logger

logger.debug(
    f'KEYCLOAK_SERVER_URL:{KEYCLOAK_SERVER_URL}, KEYCLOAK_SERVER_URL_EXT:{KEYCLOAK_SERVER_URL_EXT}, KEYCLOAK_CLIENT_ID:{KEYCLOAK_CLIENT_ID}'
)

_keycloak_instances = {}


def get_keycloak_openid(external=False) -> KeycloakOpenID:
    """Returns a singleton instance of KeycloakOpenID based on the 'external' flag."""
    if external not in _keycloak_instances:
        _keycloak_instances[external] = KeycloakOpenID(
            server_url=KEYCLOAK_SERVER_URL_EXT if external else KEYCLOAK_SERVER_URL,
            realm_name=KEYCLOAK_REALM_NAME,
            client_id=KEYCLOAK_CLIENT_ID,
            client_secret_key=KEYCLOAK_CLIENT_SECRET,
        )
    return _keycloak_instances[external]


_keycloak_admin_instances = {}


def get_keycloak_admin(external=False) -> KeycloakAdmin:
    """Returns a singleton instance of KeycloakAdmin based on the 'external' flag."""
    if external not in _keycloak_admin_instances:
        keycloak_admin = KeycloakAdmin(
            server_url=KEYCLOAK_SERVER_URL_EXT if external else KEYCLOAK_SERVER_URL,
            username='admin',
            password=KEYCLOAK_ADMIN_PASSWORD,
            realm_name='master',
            client_id='admin-cli',
            verify=True,
        )
        keycloak_admin.get_realm(KEYCLOAK_REALM_NAME)
        keycloak_admin.change_current_realm(KEYCLOAK_REALM_NAME)
        _keycloak_admin_instances[external] = keycloak_admin
    return _keycloak_admin_instances[external]
