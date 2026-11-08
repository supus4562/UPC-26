"""Analytic forward approximations and bounded importance-sampling inverse inference."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares, brentq
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from physics_predictors import summarize

LOW=np.array([np.sqrt(9.8*11),0,55,-8,-5,0.])
HIGH=np.array([129/3.6,90,125,8,5,15.])
R0=np.array([0.,0.,.11])
KD=.4*1.27*np.pi*.108**2/(2*.430)
KM=8*np.pi*1.27*.108**3/(3*.430)
GE=9.8*(1-1.27*4*np.pi*.108**3/(3*.430))


def coefficients(unit):
    speed,theta,phi,wx,wy,wz=LOW+np.asarray(unit)*(HIGH-LOW)
    theta,phi=np.deg2rad([theta,phi])
    v=speed*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    B=KM*np.array([[0,wz,-wy],[-wz,0,wx],[wy,-wx,0]])
    A=-KD*(speed*np.eye(3)+np.outer(v,v)/speed)+B
    b=KD*speed*v-np.array([0,0,GE])
    acceleration=-KD*speed*v+B@v-np.array([0,0,GE])
    return v,A,b,acceleration,A@acceleration


class Flight:
    def __init__(self,unit,method):
        self.method=method
        self.v,self.A,self.b,self.a,self.j=coefficients(unit)
        if method=='linear':
            self.ab=np.linalg.solve(self.A,self.b)
            self.e,self.V=np.linalg.eig(self.A)
            self.c=np.linalg.solve(self.V,self.v+self.ab)
            self.spectral=np.linalg.cond(self.V)<1e8

    def position(self,t):
        ts=np.atleast_1d(t).astype(float)
        if self.method=='taylor':
            r=R0+ts[:,None]*self.v+ts[:,None]**2*self.a/2+ts[:,None]**3*self.j/6
        elif self.spectral:
            r=R0+np.real((np.expm1(ts[:,None]*self.e)/self.e*self.c)@self.V.T)-ts[:,None]*self.ab
        else:
            r=np.array([R0+np.linalg.solve(self.A,(expm(self.A*s)-np.eye(3))@(self.v+self.ab))-s*self.ab for s in ts])
        return r[0] if np.ndim(t)==0 else r

    def velocity(self,t):
        if self.method=='taylor':return self.v+t*self.a+t*t*self.j/2
        return expm(self.A*t)@(self.v+self.ab)-self.ab

    def crossing(self,step=.02):
        if self.method=='taylor':
            def roots(axis,level):
                c=[self.j[axis]/6,self.a[axis]/2,self.v[axis],R0[axis]-level]
                return sorted(float(z.real) for z in np.roots(np.trim_zeros(c,'f'))
                              if abs(z.imag)<1e-8 and 0<z.real<=15)
            goals=[t for t in roots(1,11) if self.velocity(t)[1]>0]
            ground=[t for t in roots(2,.108) if self.velocity(t)[2]<0]
            if not goals:return None
            t=goals[0]
            if ground and ground[0]<=t:return None
        else:
            # Locate every bracket on a dense time grid, then refine the first event.
            times=np.r_[0.,np.arange(step,15,step),15.]
            p=self.position(times)
            g=np.flatnonzero((p[:-1,1]<11)&(p[1:,1]>=11))
            z=np.flatnonzero((p[:-1,2]>.108)&(p[1:,2]<=.108))
            if not len(g):return None
            i=g[0];t=brentq(lambda s:self.position(s)[1]-11,times[i],times[i+1],xtol=1e-11)
            if len(z):
                i=z[0];tg=brentq(lambda s:self.position(s)[2]-.108,times[i],times[i+1],xtol=1e-11)
                if tg<=t:return None
        if t<=.1:return None
        q=self.position(t)[[0,2]]
        return q if abs(q[0])<3.66 and 0<q[1]<2.44 else None


def infer(query,method,sigma=.005/np.sqrt(3),samples=4096,starts=4,seed=42,mode=False,event_step=.02):
    """Uniform unit-box prior; likelihood is never broadened. Proposal weights correct targeting."""
    query=np.asarray(query).reshape(6)
    if not np.isfinite(query).all() or sigma<=0 or samples<128 or starts<1:
        raise ValueError('Finite observations, positive sigma, samples>=128, starts>=1 required')
    rng=np.random.default_rng(seed)
    def residual(u):return (Flight(u,method).position([.05,.10]).ravel()-query)/sigma
    # Kinematic estimate only initializes fits; no true test launch parameters are read.
    r1,r2=query.reshape(2,3)
    velocity=(4*r1-r2-3*R0)/.1
    speed=np.linalg.norm(velocity)
    physical=np.array([speed,np.rad2deg(np.arccos(np.clip(velocity[2]/max(speed,1e-12),-1,1))),
                       np.rad2deg(np.arctan2(velocity[1],velocity[0])),0,0,7.5])
    initial=np.clip((physical-LOW)/(HIGH-LOW),1e-6,1-1e-6)
    fits=[]
    for i in range(starts):
        start=initial.copy()
        if i:start[3:]=rng.uniform(.01,.99,3)
        fits.append(least_squares(residual,start,bounds=(np.zeros(6),np.ones(6)),max_nfev=150))
    # Defensive mixture: 10% full prior, 90% inflated local Gaussian proposals.
    # Eigenvalue floor controls proposal spread only, NOT reported posterior covariance.
    proposals=[]
    for fit in fits:
        eig,V=np.linalg.eigh(fit.jac.T@fit.jac)
        cov=(V*(4/np.maximum(eig,4)))@V.T
        proposals.append(multivariate_normal(fit.x,cov,allow_singular=False))
    labels=rng.choice(starts+1,size=samples,p=[.1]+[.9/starts]*starts)
    draws=np.empty((samples,6))
    for k in range(starts+1):
        selected=labels==k;n=int(selected.sum())
        if n:draws[selected]=rng.random((n,6)) if k==0 else proposals[k-1].rvs(size=n,random_state=rng)
    inside=np.all((draws>=0)&(draws<=1),axis=1);draws=draws[inside]
    if not len(draws):raise ValueError('No proposals inside launch bounds')
    logq=logsumexp(np.column_stack([np.full(len(draws),np.log(.1))]+
                  [np.log(.9/starts)+p.logpdf(draws) for p in proposals]),axis=1)
    positions=[];logs=[]
    for u,lq in zip(draws,logq):
        f=Flight(u,method);q=f.crossing(event_step)
        if q is None:continue
        e=(f.position([.05,.10]).ravel()-query)/sigma
        positions.append(q);logs.append(-.5*e@e-lq)
    if not logs:raise ValueError('No goal-valid posterior proposals; increase samples/starts or inspect approximation')
    w=np.exp(np.array(logs)-logsumexp(logs));points=np.array(positions)
    report=summarize(points,w,.05 if mode else None)
    ess=float(1/(w@w))
    report.update(method=method,posterior_ess=ess,proposal_samples=samples,
                  bounded_proposals=len(draws),goal_valid_proposals=len(points),
                  best_observation_chi2=float(min(f.fun@f.fun for f in fits)),
                  optimizer_successes=sum(bool(f.success) for f in fits),
                  measurement_sigma_m=sigma,event_scan_step_s=event_step,
                  warnings=['Importance ESS below 100; rerun with more samples/starts and compare.'] if ess<100 else [],
                  uncertainty_scope='Measurement noise and launch ambiguity; excludes approximation error',
                  conditioned_on='Goal entry under approximate dynamics and bounded launch prior')
    return report,points,w
