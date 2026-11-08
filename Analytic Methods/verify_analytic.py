"""Independent numerical checks of analytical solutions and event locations."""
import numpy as np
from scipy.integrate import solve_ivp
from analytic_core import Flight,coefficients,R0,LOW,HIGH

unit=(np.array([30,80,90,0,0,0])-LOW)/(HIGH-LOW)
f=Flight(unit,'linear');v,A,b,a,j=coefficients(unit)
t=np.array([.05,.1,.3])
r=solve_ivp(lambda time,s:np.r_[s[3:],A@s[3:]+b],(0,.3),np.r_[R0,v],t_eval=t,rtol=1e-11,atol=1e-13)
np.testing.assert_allclose(f.position(t),r.y[:3].T,atol=1e-9)
ft=Flight(unit,'taylor')
np.testing.assert_allclose(ft.velocity(0),v)
np.testing.assert_allclose(ft.position(.0001),f.position(.0001),atol=1e-12)
assert ft.crossing() is not None and f.crossing() is not None
np.testing.assert_allclose(f.crossing(.02),f.crossing(.005),atol=1e-8)
# Horizontal zero-spin launch hits ground before goal in both approximations.
horizontal=(np.array([30,90,90,0,0,0])-LOW)/(HIGH-LOW)
assert Flight(horizontal,'linear').crossing() is None
assert Flight(horizontal,'taylor').crossing() is None
print('Matrix exponential, cubic launch derivatives, shared short-time expansion, event refinement and ground rejection passed.')
