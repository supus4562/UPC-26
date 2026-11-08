"""Build reusable adaptive Bayes reference and conditional Gaussian mixture from TRAIN only."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from data import load_partition


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--models',type=Path,default=Path('models'))
    p.add_argument('--components',type=int,default=16)
    p.add_argument('--min-ess',type=float,default=30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args()
    x,y,ids,meta=load_partition(a.dataset,'train')
    if not 3 <= a.min_ess < len(x) or not 1 <= a.components <= len(x) or a.threads < 1:
        p.error('Need 3 <= min ESS < training rows; valid components and threads')
    a.models.mkdir(parents=True,exist_ok=True)
    if any(a.models.iterdir()):
        p.error('Model directory must be empty; use a new --models directory')
    common=dict(train_sha256=meta['train_sha256'],test_sha256=meta['test_sha256'],train_ids=ids)
    joint=np.column_stack([x,y]); center=joint.mean(0); scale=joint.std(0);scale[scale==0]=1
    with threadpool_limits(limits=a.threads):
        g=GaussianMixture(n_components=a.components,covariance_type='full',reg_covar=1e-6,
                          random_state=a.seed,n_init=2,max_iter=500).fit((joint-center)/scale)
    if not g.converged_:
        raise RuntimeError('CGMM did not converge; no models saved. Adjust component count.')
    np.savez_compressed(a.models/'cgmm.npz',version=1,center=center,scale=scale,
                        weights=g.weights_,means=g.means_,covariances=g.covariances_,
                        converged=g.converged_,retained_shots=len(x),**common)
    np.savez_compressed(a.models/'bayes.npz',version=1,observations=x,points=y,
                        min_ess=a.min_ess,**common)
    report=dict(training_rows=len(x),components=a.components,min_ess=a.min_ess,
                cgmm_converged=bool(g.converged_),seed=a.seed,split=meta)
    (a.models/'training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
