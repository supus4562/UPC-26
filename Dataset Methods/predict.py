"""Predict using saved TRAIN-only models; never opens a dataset or fits a model."""
import argparse
import json
from pathlib import Path
import numpy as np
from models import Models
from physics_predictors import plot_crossing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=Path('models'))
    p.add_argument('--method',choices=['bayes','cgmm','both'],default='both')
    p.add_argument('--r1',type=float,nargs=3,required=True)
    p.add_argument('--r2',type=float,nargs=3,required=True)
    p.add_argument('--frame',choices=['paper','problem'],default='problem')
    p.add_argument('--output-frame',choices=['paper','problem'],default='problem')
    p.add_argument('--sigma',type=float,default=.005/np.sqrt(3))
    p.add_argument('--draws',type=int,default=10000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=Path('prediction'))
    p.add_argument('--show',action='store_true')
    a=p.parse_args(); models=Models(a.models,a.method)
    query=np.array([a.r1,a.r2])
    if a.frame=='problem':query=query[:,[1,0,2]]*[-1,1,1]
    a.output.mkdir(parents=True,exist_ok=True);reports={}
    for method in models.methods:
        r,points,w=models.predict(method,query.ravel(),a.sigma,draws=a.draws,seed=a.seed)
        plot_crossing(points,w,r,method,a.output_frame,a.output/f'goal_{method}.png',a.show)
        sign=1 if a.output_frame=='paper' else -1
        r['output_axes']=['x_right','z_up'] if sign==1 else ['y_left','z_up']
        r['mean_output_m']=(np.array(r['mean_crossing_m'])*[sign,1]).tolist()
        r['mode_output_m']=(np.array(r['approximate_mode_crossing_m'])*[sign,1]).tolist()
        transform=np.diag([sign,1])
        r['output_covariance_m2']=(transform@np.array(r['covariance_m2'])@transform).tolist()
        r['internal_report_frame']='Team 358 x_right,z_up'
        reports[method]=r
    text=json.dumps(reports,indent=2,allow_nan=False)
    (a.output/'prediction.json').write_text(text);print(text)


if __name__=='__main__':main()
