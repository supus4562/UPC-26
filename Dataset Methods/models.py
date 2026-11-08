"""Shared prediction path for CLI and benchmarks."""
from pathlib import Path
import numpy as np
import physics_predictors as core


class Models:
    def __init__(self, folder, method='both'):
        self.folder=Path(folder)
        self.methods=('bayes','cgmm') if method=='both' else (method,)
        self.saved={}
        for m in self.methods:
            with np.load(self.folder/f'{m}.npz',allow_pickle=False) as f:
                self.saved[m]={k:f[k] for k in f.files}
            if 'train_sha256' not in self.saved[m]:
                raise ValueError('Old model lacks split provenance; retrain using train.py')
        if len({str(v['train_sha256']) for v in self.saved.values()})!=1:
            raise ValueError('Models were trained on different partitions')

    def validate_test(self, meta, ids):
        for m,s in self.saved.items():
            if str(s['train_sha256'])!=meta['train_sha256'] or str(s['test_sha256'])!=meta['test_sha256']:
                raise ValueError(f'{m}: dataset does not match saved model split')
            if np.intersect1d(ids,s['train_ids']).size:
                raise ValueError('Train/test ID leakage detected')

    def predict(self, method, query, sigma, mode_bandwidth=.05, draws=10000, seed=42):
        if not np.isfinite(query).all() or not np.isfinite(sigma) or sigma<=0 or draws<1000:
            raise ValueError('Use finite query, positive sigma, and at least 1000 CGMM draws')
        s=self.saved[method]
        if method=='bayes':
            r,w=core.predict(s['observations'],s['points'],query,sigma,
                             float(s['min_ess']),True,mode_bandwidth)
            points=s['points']
        else:
            r,points,w=core.predict_cgmm(self.folder/'cgmm.npz',None,query,sigma,
                                        mode_bandwidth,draws,seed)
        r['method']='adaptive_bayes' if method=='bayes' else 'conditional_gaussian_mixture'
        return r,points,w
