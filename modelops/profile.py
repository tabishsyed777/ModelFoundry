"""Training-reference histograms and total-variation input drift checks."""
import math
import numpy as np
import pandas as pd


def build_profile(frame):
    if frame.empty or not frame.columns.is_unique or any(not isinstance(c,str) for c in frame.columns):
        raise ValueError('Reference requires rows and unique string feature names')
    features = {}
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series):
            spec = {'kind':'boolean','categories':['False','True']}
        elif pd.api.types.is_numeric_dtype(series):
            clean = series.dropna().astype(float)
            if not np.isfinite(clean).all():
                raise ValueError('Reference numbers must be finite')
            spec = {'kind':'number','thresholds':sorted(set(float(v) for v in clean.quantile([.2,.4,.6,.8]).tolist())) if len(clean) else []}
        else:
            if any(not isinstance(v,str) for v in series.dropna()):
                raise ValueError('Categorical reference values must be strings')
            counts = series.dropna().value_counts()
            categories = sorted(counts.index, key=lambda value:(-int(counts[value]),value))[:30]
            spec = {'kind':'category','categories':categories}
        spec['counts'] = bucket_counts(spec,series)
        features[name] = spec
    return {'reference_rows':len(frame),'features':features}


def bucket_counts(spec,series):
    if spec['kind']=='number':
        counts = [0]*(len(spec['thresholds'])+2)
        for value in series:
            index = len(counts)-1 if pd.isna(value) else int(np.searchsorted(spec['thresholds'],float(value),side='right'))
            counts[index] += 1
    else:
        categories = spec['categories']
        counts = [0]*(len(categories)+2)  # known categories, other, missing
        lookup = {value:index for index,value in enumerate(categories)}
        for value in series:
            index = len(counts)-1 if pd.isna(value) else lookup.get(str(value),len(categories))
            counts[index] += 1
    return counts


def summarize(spec,counts,threshold=.2):
    if len(counts)!=len(spec['counts']):
        raise ValueError('Histogram shape does not match reference')
    total, reference = sum(counts), sum(spec['counts'])
    distance = .5*sum(abs(a/total-b/reference) for a,b in zip(counts,spec['counts'])) if total else None
    return {'counts':counts,'distance':distance,'drifted':distance is not None and distance>threshold,'threshold':threshold}


def compare_profile(profile,frame,threshold=.2):
    return {name:summarize(spec,bucket_counts(spec,frame[name]),threshold) for name,spec in profile['features'].items()}


def finite_number(value):
    try:
        return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)
    except OverflowError:
        return False


def validate_rows(profile,rows):
    if not isinstance(rows,list) or not 1<=len(rows)<=1000:
        raise ValueError('Supply 1–1000 rows')
    features = profile['features']
    for row in rows:
        if not isinstance(row,dict) or set(row)!=set(features):
            raise ValueError('Every row must contain exactly the registered feature names')
        for name,spec in features.items():
            value = row[name]
            if value is None:
                continue
            kind = spec['kind']
            if kind=='number' and not finite_number(value):
                raise ValueError(f'{name} requires a finite number or null')
            if kind=='boolean' and not isinstance(value,bool):
                raise ValueError(f'{name} requires a boolean or null')
            if kind=='category' and not isinstance(value,str):
                raise ValueError(f'{name} requires a string or null')
    frame = pd.DataFrame(rows,columns=list(features))
    # Convert nulls to numpy NaN for the existing scikit-learn imputation pipelines.
    return frame.replace({None:np.nan})
