"""Shared CLI; separate entry points select the analytic approximation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from analytic_core import infer
from stream_metrics import Aggregate
from physics_predictors import plot_crossing


def common(p):
    p.add_argument('--sigma',type=float,default=.005/np.sqrt(3))
    p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--starts',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--event-step',type=float,default=.02,help='Linear-model event bracketing grid in seconds')


def validate(p,a):
    if not np.isfinite([a.sigma,a.event_step]).all() or a.sigma<=0 or not 0<a.event_step<=.1 or a.samples<128 or a.starts<1 or a.seed<0:
        p.error('Require sigma>0, 0<event-step<=0.1, samples>=128, starts>=1, seed>=0')


def worker(job):
    i,identifier,query,method,settings=job
    with threadpool_limits(limits=1):
        try:
            r,_,_=infer(query,method,**settings)
            return dict(index=int(i),id=int(identifier),report=r)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as e:
            return dict(index=int(i),id=int(identifier),error=str(e))


def evaluate_main(method):
    p=argparse.ArgumentParser(description=f'Test-only benchmark of {method} analytical approximation')
    common(p)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='0 uses full test partition')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--per-shot',action='store_true',help='Optionally write per-shot JSONL; disabled by default')
    p.add_argument('--clean',action='store_true')
    p.add_argument('--include-mode',action='store_true')
    p.add_argument('--output',type=Path,default=Path(f'metrics_{method}.json'))
    a=p.parse_args();validate(p,a)
    if a.limit<0 or a.workers<1:p.error('Invalid limit or workers')
    from data import load_partition
    x,y,ids,meta=load_partition(a.dataset,'test')
    # Identical noisy observations and subset selection to the dataset-based evaluator.
    noise=np.random.default_rng(a.seed).normal(0,a.sigma,size=x.shape)
    queries=x if a.clean else x+noise
    idx=np.arange(len(y)) if not a.limit else np.sort(np.random.default_rng(a.seed).choice(len(y),min(a.limit,len(y)),replace=False))
    settings=dict(sigma=a.sigma,samples=a.samples,starts=a.starts,mode=a.include_mode,event_step=a.event_step)
    jobs=((int(i),int(ids[i]),queries[i],method,dict(settings,seed=int(a.seed+i))) for i in idx)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    records=a.output.with_suffix('.jsonl')
    accumulator=Aggregate();failure_count=0;failure_examples=[];started=time.monotonic()
    context=ProcessPoolExecutor(a.workers) if a.workers>1 else nullcontext(None)
    # Submit at most 2*workers jobs, rather than eagerly queuing the full test set.
    def bounded(pool,jobs):
        from collections import deque
        iterator=iter(jobs);pending=deque()
        for _ in range(2*a.workers):
            job=next(iterator,None)
            if job is not None:pending.append(pool.submit(worker,job))
        while pending:
            yield pending.popleft().result()
            job=next(iterator,None)
            if job is not None:pending.append(pool.submit(worker,job))
    with context as pool, (records.open('w',encoding='utf8') if a.per_shot else nullcontext(None)) as f:
        outputs=bounded(pool,jobs) if pool else map(worker,jobs)
        for count,row in enumerate(outputs,1):
            if f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            if 'error' in row:
                failure_count+=1
                if len(failure_examples)<10:failure_examples.append(row)
            else:accumulator.add(y[row['index']],row['report'])
            print(f'{method}: {count}/{len(idx)}; failures={failure_count}',file=sys.stderr,flush=True)
    score=accumulator.finish(len(idx))
    report=dict(method=method,frame=['paper_X_right','Z_up'],split=meta,requested=len(idx),
                successful=accumulator.n,failed=failure_count,failure_examples=failure_examples,
                metrics_on_successful_queries=score,test_noise_sigma_m=0 if a.clean else a.sigma,
                settings=settings,seed=a.seed,elapsed_seconds=time.monotonic()-started,
                per_shot_records=str(records) if a.per_shot else None,
                interpretation='Errors exclude failed predictions; compare success rate too. Low ESS results are provisional.')
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2,allow_nan=False))


def predict_main(method):
    p=argparse.ArgumentParser(description=f'Original penalty prediction using {method}')
    common(p)
    p.add_argument('--r1',nargs=3,type=float,default=[1.50,.45,.44])
    p.add_argument('--r2',nargs=3,type=float,default=[2.96,.86,.74])
    p.add_argument('--frame',choices=['problem','paper'],default='problem')
    p.add_argument('--output-frame',choices=['problem','paper'],default='problem')
    p.add_argument('--output',type=Path,default=Path(f'prediction_{method}'))
    p.add_argument('--show',action='store_true')
    a=p.parse_args();validate(p,a)
    d=np.array([a.r1,a.r2])
    if a.frame=='problem':d=d[:,[1,0,2]]*[-1,1,1]
    with threadpool_limits(limits=1):
        r,points,w=infer(d.ravel(),method,a.sigma,a.samples,a.starts,a.seed,True,a.event_step)
    a.output.mkdir(parents=True,exist_ok=True)
    plot_crossing(points,w,r,method,a.output_frame,a.output/'goal.png',a.show)
    sign=1 if a.output_frame=='paper' else -1
    r['output_axes']=['X_right','Z_up'] if sign==1 else ['y_left','z_up']
    r['mean_output_m']=(np.array(r['mean_crossing_m'])*[sign,1]).tolist()
    r['mode_output_m']=(np.array(r['approximate_mode_crossing_m'])*[sign,1]).tolist()
    D=np.diag([sign,1]);r['output_covariance_m2']=(D@np.array(r['covariance_m2'])@D).tolist()
    r['internal_report_frame']='Team 358 X_right,Z_up'
    text=json.dumps(r,indent=2,allow_nan=False);(a.output/'prediction.json').write_text(text);print(text)
