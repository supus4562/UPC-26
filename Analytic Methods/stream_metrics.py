"""Streaming benchmark accumulator; retains only scalars needed for exact medians."""
import numpy as np

class Aggregate:
    def __init__(self):
        self.n=0; self.sse=np.zeros(2);self.radial=0.;self.mode_sse=np.zeros(2);self.mode_radial=0.;self.mode_n=0
        self.marginal=np.zeros(2);self.width=np.zeros(2);self.radial_hits=0;self.ellipses=0;self.hits=0;self.area=0.
        self.keys=['sigma_horizontal_m','sigma_vertical_m','radial_rms_uncertainty_m','radial_95_radius_m']
        self.values={k:[] for k in self.keys};self.ess=[]
    def add(self,actual,r):
        self.n+=1
        delta=np.asarray(r['mean_crossing_m'])-actual
        self.sse+=delta**2;self.radial+=np.linalg.norm(delta)
        if r['approximate_mode_crossing_m'] is not None:
            e=np.asarray(r['approximate_mode_crossing_m'])-actual
            self.mode_sse+=e**2;self.mode_radial+=np.linalg.norm(e);self.mode_n+=1
        for k in self.keys:self.values[k].append(float(r[k]))
        self.ess.append(float(r['posterior_ess']))
        intervals=np.array(list(r['marginal_95_percent_m'].values()))
        self.width+=intervals[:,1]-intervals[:,0]
        self.marginal+=(actual>=intervals[:,0])&(actual<=intervals[:,1])
        self.radial_hits+=np.linalg.norm(delta)<=r['radial_95_radius_m']
        e=r['joint_95_percent_ellipse']
        if e and min(e['semi_axes_m'])>0:
            major=np.array(e['major_axis_unit_vector']);minor=np.array([-major[1],major[0]])
            d=actual-np.array(e['center'])
            self.hits+=np.sum((np.array([d@major,d@minor])/e['semi_axes_m'])**2)<=1
            self.ellipses+=1;self.area+=np.pi*np.prod(e['semi_axes_m'])
    def finish(self,requested):
        if not self.n:return None
        def score(sse,radial,n):
            mse=sse/n
            return dict(mse_horizontal_m2=float(mse[0]),mse_vertical_m2=float(mse[1]),
                mse_per_coordinate_m2=float(mse.mean()),radial_mse_m2=float(mse.sum()),
                rmse_horizontal_m=float(np.sqrt(mse[0])),rmse_vertical_m=float(np.sqrt(mse[1])),
                radial_rmse_m=float(np.sqrt(mse.sum())),mean_radial_error_m=float(radial/n))
        uncertainty={k:dict(mean=float(np.mean(v)),median=float(np.median(v))) for k,v in self.values.items()}
        uncertainty['marginal_95_width_m']=dict(zip(['horizontal','vertical'],(self.width/self.n).tolist()))
        result=dict(evaluated=self.n,mean_crossing=score(self.sse,self.radial,self.n),uncertainty=uncertainty,
            marginal_95_coverage=dict(zip(['horizontal','vertical'],(self.marginal/self.n).tolist())),
            radial_95_coverage=float(self.radial_hits/self.n),
            ellipse_95=dict(available=self.ellipses,availability_rate=self.ellipses/self.n,covered=int(self.hits),
                coverage_when_available=float(self.hits/self.ellipses) if self.ellipses else None,
                coverage_all_queries=float(self.hits/self.n),coverage_all_requested=float(self.hits/requested),
                mean_area_m2=float(self.area/self.ellipses) if self.ellipses else None),
            posterior_sampling=dict(median_ess=float(np.median(self.ess)),minimum_ess=min(self.ess),
                fraction_ess_below_100=float(np.mean(np.array(self.ess)<100))))
        if self.mode_n:result['mode_crossing']=score(self.mode_sse,self.mode_radial,self.mode_n)
        return result
