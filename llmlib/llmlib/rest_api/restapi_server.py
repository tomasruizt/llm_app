import torch
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from llmlib.bundler import Bundler
from llmlib.bundler_request import BundlerRequest
from llmlib.rest_api.restapi_client import RequestDto, to_bundler_msg
from llmlib.runtime import filled_model_registry


import os

import bugsnag
from bugsnag.asgi import BugsnagMiddleware


def create_fastapi_app() -> FastAPI:
    bugsnag.configure(api_key=os.environ["BUGSNAG_API_KEY"])

    bundler = Bundler(registry=filled_model_registry())
    app = FastAPI()
    app.add_middleware(BugsnagMiddleware)

    header = APIKeyHeader(name="X-API-Key")

    def is_authorized(api_key: str = Security(header)) -> bool:
        if api_key != os.environ["LLMS_REST_API_KEY"]:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        return True

    @app.get("/models/")
    def _(_=Depends(is_authorized)):
        return bundler.registry.all_model_ids()

    @app.post("/completion/")
    def _(req: RequestDto, _=Depends(is_authorized)):
        breq = BundlerRequest(
            model_id=req.model, msgs=[to_bundler_msg(msg) for msg in req.msgs]
        )
        return {"response": bundler.get_response(breq)}

    @app.post("/clear-gpu/")
    def _(_=Depends(is_authorized)):
        bundler.clear_model_on_gpu()
        return {"status": "success"}

    @app.exception_handler(torch.cuda.OutOfMemoryError)
    def _(req, exc):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Error. GPU out of memory. There might be another workload running on the GPU."
            },
        )

    return app
