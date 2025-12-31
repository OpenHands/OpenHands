from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        if scope.get("type") != "http":
            # Return 404 for non-http scopes (e.g., websocket)
            return Response("Not Found", status_code=404)
        
        # Don't serve index.html for API requests
        if path.startswith('/api/'):
            return Response("Not Found", status_code=404)
        
        try:
            return await super().get_response(path, scope)
        except Exception:
            # Only serve index.html for non-API paths
            if not path.startswith('/api/'):
                return await super().get_response('index.html', scope)
            raise
