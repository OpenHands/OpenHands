# Legacy V0 health endpoints - now delegate to app_server
from fastapi import FastAPI


def add_health_endpoints(app: FastAPI):
    # Import and include the health routes from app_server
    from openhands.app_server.routes.health_routes import router as health_router
    app.include_router(health_router)
