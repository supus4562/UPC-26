"""Benchmark saved models on the fixed TEST partition. No fitting or resplitting."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from data import load_partition
from models import Models


def point_metrics(truth, estimates):
    error=np.asarray(estimates)-truth
    mse=np.mean(error**2,axis=0)
    return dict(mse_horizontal_m2=float(mse[0]),mse_vertical_m2=float(mse[1]),
                mse_per_coordinate_m2=float(mse.mean()),radial_mse_m2=float(mse.sum()),
                rmse_horizontal_m=float(np.sqrt(mse[0])),rmse_vertical_m=float(np.sqrt(mse[1])),
                radial_rmse_m=float(np.sqrt(mse.sum())),
                mean_radial_error_m=float(np.linalg.norm(error,axis=1).mean()))


def benchmark(truth,reports):
    means=np.array([r['mean_crossing_m'] for r in reports])
    uncertainty={k:dict(mean=float(np.mean([r[k] for r in reports])),
                        median=float(np.median([r[k] for r in reports])))
                 for k in ['sigma_horizontal_m','sigma_vertical_m',
                           'radial_rms_uncertainty_m','radial_95_radius_m']}
    uncertainty['marginal_95_width_m'] = dict(zip(['horizontal','vertical'],
        np.mean([[v[1]-v[0] for v in r['marginal_95_percent_m'].values()] for r in reports],axis=0).tolist()))
    marginal=[];radial=[];ellipse=[];areas=[]
    for actual,r in zip(truth,reports):
        intervals=np.array(list(r['marginal_95_percent_m'].values()))
        marginal.append((actual>=intervals[:,0]) & (actual<=intervals[:,1]))
        radial.append(np.linalg.norm(actual-np.array(r['mean_crossing_m']))<=r['radial_95_radius_m'])
        e=r['joint_95_percent_ellipse']
        if e and min(e['semi_axes_m'])>0:
            major=np.array(e['major_axis_unit_vector']);minor=np.array([-major[1],major[0]])
            delta=actual-np.array(e['center'])
            ellipse.append(float(np.sum((np.array([delta@major,delta@minor])/e['semi_axes_m'])**2))<=1)
            areas.append(float(np.pi*np.prod(e['semi_axes_m'])))
    result=dict(evaluated=len(reports),mean_crossing=point_metrics(truth,means),uncertainty=uncertainty,
                marginal_95_coverage=dict(horizontal=float(np.mean(marginal,axis=0)[0]),
                                          vertical=float(np.mean(marginal,axis=0)[1])),
                radial_95_coverage=float(np.mean(radial)),
                ellipse_95=dict(available=len(ellipse),availability_rate=len(ellipse)/len(reports),
                    covered=int(sum(ellipse)),coverage_when_available=float(np.mean(ellipse)) if ellipse else None,
                    coverage_all_queries=sum(ellipse)/len(reports),
                    mean_area_m2=float(np.mean(areas)) if areas else None))
    if reports[0]['approximate_mode_crossing_m'] is not None:
        result['mode_crossing']=point_metrics(truth,[r['approximate_mode_crossing_m'] for r in reports])
    if 'effective_sample_size' in reports[0]:
        result['adaptive_support']={k:dict(median=float(np.median([r[k] for r in reports])),
                                           maximum=float(np.max([r[k] for r in reports])))
                  for k in ['measurement_only_ess','effective_sample_size','effective_sigma_m']}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--models',type=Path,default=Path('models'))
    p.add_argument('--method',choices=['both','bayes','cgmm'],default='both')
    p.add_argument('--limit',type=int,default=0,help='0 evaluates all test rows; otherwise fixed random subset')
    p.add_argument('--sigma',type=float,default=.005/np.sqrt(3))
    p.add_argument('--clean',action='store_true',help='Disable injected noise (likelihood sigma still applies)')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--draws',type=int,default=10000)
    p.add_argument('--include-mode',action='store_true',help='Also benchmark KDE mode; substantially slower')
    p.add_argument('--output',type=Path,default=Path('metrics.json'))
    a=p.parse_args()
    if a.limit<0 or a.draws<1000 or not np.isfinite(a.sigma) or a.sigma<=0:p.error('Invalid limit, draws or sigma')
    x,y,ids,meta=load_partition(a.dataset,'test');models=Models(a.models,a.method);models.validate_test(meta,ids)
    # Generate identical noise for every method, independent of optional subset size.
    noise=np.random.default_rng(a.seed).normal(0,a.sigma,size=x.shape)
    queries=x if a.clean else x+noise
    idx=np.arange(len(y)) if not a.limit else np.sort(np.random.default_rng(a.seed).choice(len(y),min(a.limit,len(y)),replace=False))
    result=dict(frame=['paper_x_right','z_up'],training_rows=meta['train_rows'],test_partition_rows=len(y),
                evaluated_rows=len(idx),split=meta,seed=a.seed,observation_noise='none' if a.clean else 'independent Gaussian',
                test_noise_sigma_m=0 if a.clean else a.sigma,likelihood_sigma_m=a.sigma,
                cgmm_posterior_draws=a.draws,conditioned_on='airborne goal entry',methods={})
    with threadpool_limits(limits=1):
        for method in models.methods:
            reports=[]
            for count,i in enumerate(idx,1):
                r,_,_=models.predict(method,queries[i],a.sigma,.05 if a.include_mode else None,a.draws,int(a.seed+i))
                reports.append(r)
                if count%50==0:print(f'{method}: {count}/{len(idx)}',file=sys.stderr)
            result['methods'][method]=benchmark(y[idx],reports)
            result['methods'][method]['point_estimator']='posterior mean (primary MSE estimator)'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
