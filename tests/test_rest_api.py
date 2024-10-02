from fastapi.testclient import TestClient
from llmlib.bundler_request import BundlerRequest
import llmlib.rest_api.restapi_client as llmclient
from llmlib.rest_api.restapi_server import create_fastapi_app
from llmlib.phi3.phi3 import Phi3Vision
import pytest
from .helpers import is_ci, mona_lisa_message


def app():
    return TestClient(create_fastapi_app())


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_rest_api_get_completion():
    breq: BundlerRequest = _mona_lisa_request()
    response = llmclient.get_completion_from_rest_api(source=app(), breq=breq)
    assert response.status_code == 200, response.content
    assert "portrait" in response.json()["response"].lower()


def test_rest_api_get_models():
    response = llmclient.get_models(source=app())
    assert response.status_code == 200, response.content
    assert len(response.json()) > 3


@pytest.mark.skip(reason="This test requires the REST API to be running")
def test_rest_api_integration_test():
    breq: BundlerRequest = _mona_lisa_request()
    response = llmclient.get_completion_from_rest_api(breq)
    llmclient.clear_gpu()
    assert response.status_code == 200, response.content
    assert "portrait" in response.json()["response"].lower()


def _mona_lisa_request() -> BundlerRequest:
    msg = mona_lisa_message()
    some_valid_modelid: str = Phi3Vision.model_id
    return BundlerRequest(model_id=some_valid_modelid, msgs=[msg])
