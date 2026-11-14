"""Local trusted-artifact registry and transactional aggregate monitoring."""
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4
import joblib
import numpy as np
import pandas as pd
import sklearn
from .profile import build_profile, summarize


def runtime_versions():
    return {'python':platform.python_version(),'scikit_learn':sklearn.__version__,
            'numpy':np.__version__,'pandas':pd.__version__,'joblib':joblib.__version__}


class Registry:
    def __init__(self,directory):
        self.directory = Path(directory)
        (self.directory/'artifacts').mkdir(parents=True,exist_ok=True)
        self.database = self.directory/'registry.sqlite3'
        with self.connect() as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS versions (version TEXT PRIMARY KEY, manifest TEXT NOT NULL, profile TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS monitoring (version TEXT PRIMARY KEY, requests INTEGER NOT NULL, rows INTEGER NOT NULL, latency_sum REAL NOT NULL, latencies TEXT NOT NULL, counts TEXT NOT NULL)')

    def connect(self):
        return sqlite3.connect(self.database,timeout=15)

    def register(self,model,reference,metadata):
        if list(model.feature_names_in_) != list(reference.columns):
            raise ValueError('Reference columns must match the fitted pipeline')
        profile = build_profile(reference)
        version = uuid4().hex
        buffer = io.BytesIO()
        joblib.dump(model,buffer,compress=3)
        artifact = buffer.getvalue()
        manifest = {'version':version,'created_at':datetime.now(timezone.utc).isoformat(),
                    'sha256':hashlib.sha256(artifact).hexdigest(),'runtime':runtime_versions(),
                    'metadata':metadata,'features':{name:spec['kind'] for name,spec in profile['features'].items()},
                    'reference_rows':len(reference)}
        manifest_json = json.dumps(manifest,allow_nan=False)
        artifact_path = self.directory/'artifacts'/f'{version}.joblib'
        try:
            with artifact_path.open('xb') as stream:
                stream.write(artifact)
            with self.connect() as connection:
                connection.execute('INSERT INTO versions VALUES (?,?,?)',(version,manifest_json,json.dumps(profile,allow_nan=False)))
                counts = {name:[0]*len(spec['counts']) for name,spec in profile['features'].items()}
                connection.execute('INSERT INTO monitoring VALUES (?,0,0,0,?,?)',(version,'[]',json.dumps(counts)))
        except Exception:
            artifact_path.unlink(missing_ok=True)
            raise
        return manifest

    def describe(self,version):
        if not re.fullmatch('[a-f0-9]{32}',version):
            raise KeyError(version)
        with self.connect() as connection:
            row = connection.execute('SELECT manifest,profile FROM versions WHERE version=?',(version,)).fetchone()
        if row is None:
            raise KeyError(version)
        return json.loads(row[0]),json.loads(row[1])

    def list_versions(self):
        with self.connect() as connection:
            return [json.loads(row[0]) for row in connection.execute('SELECT manifest FROM versions ORDER BY rowid DESC')]

    def load(self,version):
        manifest,profile = self.describe(version)
        if manifest['runtime'] != runtime_versions():
            raise ValueError('Model runtime differs from registration; retrain with this runtime')
        try:
            artifact = (self.directory/'artifacts'/f'{version}.joblib').read_bytes()
        except OSError:
            raise ValueError('Registered artifact is unavailable') from None
        if hashlib.sha256(artifact).hexdigest()!=manifest['sha256']:
            raise ValueError('Artifact checksum mismatch')
        # joblib is executable serialization: registry artifacts must be operator-owned and trusted.
        try:
            model = joblib.load(io.BytesIO(artifact))
        except Exception as error:
            raise ValueError('Registered artifact cannot be loaded') from error
        return model,manifest,profile

    def record(self,version,counts,latency_ms,rows):
        self.describe(version)
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            current = connection.execute('SELECT requests,rows,latency_sum,latencies,counts FROM monitoring WHERE version=?',(version,)).fetchone()
            prior = json.loads(current[4])
            merged = {name:[a+b for a,b in zip(prior[name],values)] for name,values in counts.items()}
            latencies = (json.loads(current[3])+[latency_ms])[-100:]
            connection.execute('UPDATE monitoring SET requests=?, rows=?, latency_sum=?, latencies=?, counts=? WHERE version=?',
                               (current[0]+1,current[1]+rows,current[2]+latency_ms,json.dumps(latencies),json.dumps(merged),version))

    def monitor(self,version):
        _,profile = self.describe(version)
        with self.connect() as connection:
            row = connection.execute('SELECT requests,rows,latency_sum,latencies,counts FROM monitoring WHERE version=?',(version,)).fetchone()
        latencies,counts = json.loads(row[3]),json.loads(row[4])
        return {'version':version,'requests':row[0],'rows':row[1],
                'latency_ms':{'mean':row[2]/row[0] if row[0] else None,
                              'p50':float(np.percentile(latencies,50)) if latencies else None,
                              'p95':float(np.percentile(latencies,95)) if latencies else None,
                              'percentile_window_requests':len(latencies)},
                'features':{name:summarize(spec,counts[name]) for name,spec in profile['features'].items()},
                'scope':'Input histograms across all successful requests; no ground-truth accuracy measurement.'}
