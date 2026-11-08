"""Identical aggregate metric definitions to the dataset pipeline."""
import numpy as np

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

