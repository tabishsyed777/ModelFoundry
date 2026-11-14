"""Train and register local artifacts through the existing ModelFoundry trainer."""
import argparse
import json
from pathlib import Path
import pandas as pd
from modelfoundry import train_models
from .registry import Registry


def main():
    parser = argparse.ArgumentParser(description='Train and register a ModelFoundry pipeline')
    parser.add_argument('--csv',required=True)
    parser.add_argument('--target',required=True)
    parser.add_argument('--task',choices=['classification','regression'],required=True)
    parser.add_argument('--algorithm',required=True)
    parser.add_argument('--test-fraction',type=float,default=.25)
    parser.add_argument('--registry',default='runs/registry')
    args = parser.parse_args()
    try:
        data = pd.read_csv(args.csv)
        if args.target not in data or not 0<args.test_fraction<1:
            raise ValueError('Provide a valid target and test fraction between zero and one')
        metrics,result = train_models(data,args.target,args.task,[args.algorithm],args.test_fraction)
        manifest = Registry(args.registry).register(result['models'][args.algorithm],result['X_train'],
            {'algorithm':args.algorithm,'task':args.task,'target':args.target,
             'holdout_metrics':metrics.iloc[0].to_dict(),'train_rows':result['train_rows'],'test_rows':result['test_rows']})
        print(json.dumps(manifest,indent=2,allow_nan=False))
    except (ValueError,KeyError,OSError) as error:
        parser.exit(2,f'Registration failed: {error}\n')


if __name__=='__main__':
    main()
