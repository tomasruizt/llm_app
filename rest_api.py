from fastapi import FastAPI
from llmlib.rest_api.restapi_server import create_fastapi_app

app: FastAPI = create_fastapi_app()
