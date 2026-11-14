"""Local prediction API. Artifact registration stays outside HTTP."""
import os
import json
from secrets import compare_digest
import time
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict
from .profile import compare_profile, validate_rows
from .registry import Registry


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    rows: list[dict]


def create_app(directory=None,api_key=None):
    # Initialize the registry lazily so importing this module has no filesystem side effects.
    directory = directory or os.getenv('MODELFOUNDRY_REGISTRY','runs/registry')
    api_key = api_key if api_key is not None else os.getenv('MODELFOUNDRY_API_KEY')
    app = FastAPI(title='ModelFoundry Deployment & Monitoring',version='0.1.0')
    bearer = HTTPBearer(auto_error=False)
    def registry():
        return Registry(directory)
    def authorize(credentials:HTTPAuthorizationCredentials | None = Depends(bearer)):
        if api_key and (credentials is None or not compare_digest(credentials.credentials,api_key)):
            raise HTTPException(401,'Valid bearer token required',headers={'WWW-Authenticate':'Bearer'})
    def describe(version):
        try:
            return registry().describe(version)
        except KeyError:
            raise HTTPException(404,'Unknown model version') from None

    @app.get('/health')
    def health():
        return {'status':'ok','service':'modelfoundry-modelops'}

    @app.get('/models',dependencies=[Depends(authorize)])
    def models():
        return registry().list_versions()

    @app.get('/models/{version}',dependencies=[Depends(authorize)])
    def model(version:str):
        return describe(version)[0]

    @app.get('/monitor/{version}',dependencies=[Depends(authorize)])
    def monitor(version:str):
        describe(version)
        return registry().monitor(version)

    @app.post('/predict/{version}',dependencies=[Depends(authorize)])
    def predict(version:str,request:PredictionRequest):
        started = time.perf_counter()
        _,profile = describe(version)
        try:
            frame = validate_rows(profile,request.rows)
        except ValueError as error:
            raise HTTPException(422,str(error)) from None
        try:
            model,manifest,profile = registry().load(version)
        except ValueError:
            raise HTTPException(503,'Model artifact unavailable, incompatible, or failed integrity checks') from None
        try:
            predictions = model.predict(frame).tolist()
            json.dumps(predictions,allow_nan=False)
        except Exception:
            raise HTTPException(422,'Registered pipeline could not predict these inputs') from None
        latency_ms = (time.perf_counter()-started)*1000
        comparison = compare_profile(profile,frame)
        registry().record(version,{name:item['counts'] for name,item in comparison.items()},latency_ms,len(frame))
        return {'version':manifest['version'],'predictions':predictions,'latency_ms':latency_ms,'batch_drift':comparison}
    return app


app = create_app()
