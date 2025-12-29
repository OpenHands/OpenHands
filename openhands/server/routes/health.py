from fastapi import FastAPI

from openhands.runtime.utils.system_stats import get_system_info


def add_health_endpoints(app: FastAPI):
    @app.get('/alive')
    async def alive():
        return {'status': 'ok'}

    @app.get('/health')
    async def health() -> str:
        return 'OK'

    @app.get('/server_info')
    async def get_server_info():
        return get_system_info()
