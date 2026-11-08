"""Verify Team 358 equations, coordinate conventions, events and storage."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.integrate import solve_ivp
import generate
from generate import integrate, simulate, SCHEMA, hit_goal_plane

config = dict(timeout=15, rtol=1e-8, atol=1e-10, max_step=.02,
              speed_min=10.38, speed_max=35.8333333333, polar_max=90,
              azimuth_min=55, azimuth_max=125, spin_min=[-8,-5,0], spin_max=[8,5,15],
              whole_ball=False)

# Inspect the actual RHS passed to solve_ivp against Eq. 10, all spin signs.
with patch.object(generate, "solve_ivp") as solver:
    integrate([3,30,6], [-8,5,15], config)
    rhs = solver.call_args.args[0]
    velocity=np.array([3.,30.,6.])
    omega=np.array([-8.,5.,15.])
    mass,radius,rho=.43,.108,1.27
    kd=.4*rho*np.pi*radius**2/(2*mass)
    km=8*np.pi*rho*radius**3/(3*mass)
    ge=9.8*(1-rho*4*np.pi*radius**3/(3*mass))
    expected=-kd*np.linalg.norm(velocity)*velocity+km*np.cross(velocity,omega)-[0,0,ge]
    np.testing.assert_allclose(rhs(0,[0,0,.11,*velocity])[3:], expected)

# Check initial conditions directly, independently of whether the shot scores.
for theta,phi,expected in [(90,90,[0,1,0]),(0,90,[0,0,1]),
                           (90,55,[np.cos(np.deg2rad(55)),np.sin(np.deg2rad(55)),0]),(90,125,[np.cos(np.deg2rad(125)),np.sin(np.deg2rad(125)),0])]:
    with patch.object(generate, "integrate") as solver:
        solver.return_value.success=False
        simulate((0,np.array([.5,theta/90,(phi-55)/70,0,.5,1]),config))
        velocity,omega,_=solver.call_args.args
        np.testing.assert_allclose(velocity/np.linalg.norm(velocity),expected,atol=1e-14)
        np.testing.assert_allclose(omega,[-8,0,15])

for omega in ([0,0,0],[-8,5,15],[8,-5,-15]):
    a=integrate([0,30,6],omega,config)
    b=integrate([0,30,6],omega,dict(config,rtol=1e-11,atol=1e-13,max_step=.005))
    np.testing.assert_allclose(a.y,b.y,atol=1e-6,rtol=0)
    np.testing.assert_allclose(a.y_events[0],b.y_events[0],atol=1e-6,rtol=0)
    assert abs(a.y_events[0][0,1]-11)<1e-10

def gravity(t,s):
    return [*s[3:],0,0,-9.8]
ballistic=solve_ivp(gravity,(0,1),[0,0,.11,3,30,6],events=hit_goal_plane,
                    rtol=1e-11,atol=1e-13)
t=11/30
np.testing.assert_allclose(ballistic.y_events[0][0,:3],
                           [3*t,11,.11+6*t-4.9*t*t],atol=1e-10)
assert integrate([0,10,0],[0,0,0],config).t_events[1].size==1
unit=np.array([.7,80/90,.5,.5,.5,0])
status,row=simulate((0,unit,config))
assert status=="accepted",status
status,_=simulate((1,np.array([.7,80/90,0,.5,.5,0]),config))
assert status=="outside_goal",status
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/"test.parquet"
    pq.write_table(pa.Table.from_pylist([row],schema=SCHEMA),path)
    assert pq.read_table(path).to_pylist()==[row]
print("Team 358 RHS, launch angles, signed spins, events, refinement and Parquet checks passed.")

# Pipeline invariants and known metric values, independent of fitted model quality.
from data import create_split, load_partition
from evaluate import point_metrics
from physics_predictors import summarize
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);source=root/'raw';source.mkdir()
    rows=[dict(row,id=i) for i in range(101)]
    pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),source/'part-000.parquet')
    meta=create_split(source,root/'split')
    _,_,train_ids,_=load_partition(root/'split','train')
    _,_,test_ids,_=load_partition(root/'split','test')
    assert len(train_ids)==80 and len(test_ids)==21
    assert not np.intersect1d(train_ids,test_ids).size
    assert create_split(source,root/'split')==meta
    path=root/'split'/'test'/'part-00000.parquet'
    changed=pq.read_table(path).to_pylist();changed[0]['z_goal']+=.01
    pq.write_table(pa.Table.from_pylist(changed,schema=SCHEMA),path)
    try:load_partition(root/'split','test')
    except ValueError:pass
    else:raise AssertionError('Modified partition was not rejected')
scores=point_metrics(np.zeros((2,2)),np.array([[3,4],[-3,-4]]))
assert scores['radial_mse_m2']==25 and scores['radial_rmse_m']==5
r=summarize(np.array([[1.,2.],[1.,-2.],[-1.,2.],[-1.,-2.]]),np.ones(4)/4,None)
np.testing.assert_allclose([r['sigma_horizontal_m'],r['sigma_vertical_m'],r['radial_rms_uncertainty_m']], [1,2,np.sqrt(5)])
assert r['joint_95_percent_ellipse'] is not None
print('Split disjointness, stable resume, fingerprint rejection, MSE and uncertainty checks passed.')
