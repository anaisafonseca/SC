import numpy as np
import scipy.signal as sgn
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import control as ctl

def graph_size(n):
    """Função auxiliar para definir tamanho dos gráficos"""
    return (n*(1+5**0.5)/2, n)

#  v    = vehicle velocity (m/s)
#  t    = time (sec)
#  u    = gas pedal position (-50% to 100%)
Cd = 0.24  # drag coefficient
dens = 1.23  # air density (kg/m^3)
A = 5.0  # cross-sectional area (m^2)
Fp = 30.  # thrust parameter (N/%pedal)
m = 700.  # vehicle mass (kg)


t = np.arange(start=0, stop=60, step=0.1)

count = np.arange(start=0, stop=599, step=1)

u = np.zeros(t.shape)

u[t>=0] = 0
u[t>=5] = 0.5
u[t>=10] = 0.8
u[t>=12] = 0
u[t>=25] = 1
u[t>=40] = 0.5
u[t>=50] = 0.8
u[t>=60] = 0

v1 = ctl.tf([Fp*0,0], [m, 0, (dens * A * Cd)])
v2 = ctl.tf([Fp*0.5,0], [m, 0, (dens * A * Cd)])
v3 = ctl.tf([Fp*0.8,0], [m, 0, (dens * A * Cd)])
v4 = ctl.tf([Fp*1,0], [m, 0, (dens * A * Cd)])



ts, sr1 = ctl.step_response(v1, t)
ts, sr2 = ctl.step_response(v2, t)
ts, sr3 = ctl.step_response(v3, t)
ts, sr4 = ctl.step_response(v4, t)

sr = np.zeros(t.shape)

sr[0:50] = sr1[0:50]
sr[50:100] = sr2[50:100]
sr[100:120] = sr3[100:120]
sr[120:250] = sr1[120:250]
sr[250:400] = sr4[250:400]
sr[400:500] = sr2[400:500]
sr[500:600] = sr3[500:600]

s = ctl.TransferFunction.s
erro = np.zeros(t.shape)

for i in count:
    erro[int(i)] = u[int(i)] - sr[int(i)]

#print(erro)

Kp = 2.
Ki = 2.
Kd = 0.5

pid = Kp + Ki/s + Kd*s

sys_ctl1 = ctl.feedback(pid*v1)
sys_ctl2 = ctl.feedback(pid*v2)
sys_ctl3 = ctl.feedback(pid*v3)
sys_ctl4 = ctl.feedback(pid*v4)

ts_ctrl, sr1_ctrl = ctl.step_response(sys_ctl1, t)
ts_ctrl, sr2_ctrl = ctl.step_response(sys_ctl2, t)
ts_ctrl, sr3_ctrl = ctl.step_response(sys_ctl3, t)
ts_ctrl, sr4_ctrl = ctl.step_response(sys_ctl4, t)

sys = np.zeros(t.shape)

sys[0:50] = sr1_ctrl[0:50]
sys[50:100] = sr2_ctrl[50:100]
sys[100:120] = sr3_ctrl[100:120]
sys[120:250] = sr1_ctrl[120:250]
sys[250:400] = sr4_ctrl[250:400]
sys[400:500] = sr2_ctrl[400:500]
sys[500:600] = sr3_ctrl[500:600]

print(sys)



fig = plt.figure(figsize=graph_size(7))
plt.plot(t, sr, label='transferência', color='b', alpha=0.5, linewidth=3)
plt.plot(t, sys, label='controlado', color='r', alpha=0.5, linewidth=3)
plt.plot(t, u, label='original', color='g', alpha=0.5, linewidth=3)
plt.grid('on')
plt.xlabel('Tempo [s]')
plt.legend()
plt.show()


