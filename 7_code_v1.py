from fipy.meshes import Grid3D
from fipy import *
from fipy.tools import numerix
from fipy.terms.implicitDiffusionTerm import ImplicitDiffusionTerm
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
from fipy import Viewer
import pdb
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

#from pypower.api import case9, ppoption, runpf, printpf
# cooperative dynamics model parameters
alpha_A = 5.0
alpha_B = 6.0
beta = 0.5
ksiA = 1.0
ksiB = 3.0
RA = alpha_A/beta
RB = alpha_B/beta
D = 1

# power network model
def cost_func(x, G, P_D, a, b, c, rateA, PMIN, PMAX):
	P_T = []
	#P_T_max = 0.1
	#P_T_min = -0.1
	P_local_max = 0.03
	P_local_min = -0.03
	Num_bus = len(P_D)
	thet = x[:Num_bus]
	eps = x[Num_bus:]
	for n, nbrs in G.adj.items():
		P_T.append(P_D[n])
		for nbr, eattr in nbrs.items():
			bij = eattr['bij']
			gij = eattr['gij']
			rate_ij = eattr['rateA']
			P_ij = bij*(thet[n] - thet[nbr]) + gij*(eps[n] - eps[nbr]) 
			if P_ij > rate_ij:
				P_ij = rate_ij
			elif P_ij < -rate_ij:
				P_ij = -rate_ij
			P_T[n] += P_ij
	P_T = np.array(P_T)
	P_T[(P_T - PMAX>0)] = PMAX[(P_T - PMAX>0)]
	P_T[(P_T - PMIN<0)] = PMIN[(P_T - PMIN<0)]
	return np.sum(a*P_T**2 + b*P_T + c)

def output_P_T(x, G, P_D, a, b, c, rateA, PMIN, PMAX):
    P_T = []
        #P_T_max = 0.1
        #P_T_min = -0.1
    P_local_max = 0.03
    P_local_min = -0.03
    Num_bus = len(P_D)
    thet = x[:Num_bus]
    eps = x[Num_bus:]
    for n, nbrs in G.adj.items():
        P_T.append(P_D[n])
        for nbr, eattr in nbrs.items():
                bij = eattr['bij']
                gij = eattr['gij']
                rate_ij = eattr['rateA']
                P_ij = bij*(thet[n] - thet[nbr]) + gij*(eps[n] - eps[nbr])
                if P_ij > rate_ij:
                        P_ij = rate_ij
                elif P_ij < -rate_ij:
                        P_ij = -rate_ij
                P_T[n] += P_ij
    P_T = np.array(P_T)
    P_T[(P_T - PMAX>0)] = PMAX[(P_T - PMAX>0)]
    P_T[(P_T - PMIN<0)] = PMIN[(P_T - PMIN<0)]
    return P_T

Num_bus = 30
AA = range(24)
BB = range(24)
CC = range(24)
PMAX = range(24)
PMIN = range(24)
#MATPOWER
indices = range(Num_bus) 
gen_inds = np.array([0, 1, 12, 21, 22, 26])
As = np.array([0.02, 0.0175, 0.0625, 0.00834, 0.025 ,0.025])
Bs = np.array([2, 1.75, 1, 3.25, 3, 3])
Cs = np.array([0, 0, 0, 0, 0, 0])
Pmax = np.array([0.80, 0.80, 0.50, 0.55, 0.30, 0.40])
Pmin = np.array([0, 0, 0, 0, 0, 0])

a = np.zeros(Num_bus)
b = np.zeros(Num_bus)
c = np.zeros(Num_bus)
PMAX = np.zeros(Num_bus)
PMIN = np.zeros(Num_bus)

j = 0
for gid in gen_inds:
	a[gid] = As[j]
	b[gid] = Bs[j]
	c[gid] = Cs[j]
	PMAX[gid] = Pmax[j]
	PMIN[gid] = Pmin[j]
	j += 1

P_D = np.array([ 0, 0.2170, 0.0240, 0.0760, 0, 0, 0.2280, 0.3000, 0, 0.0580, 0, 0.1120, 0, 0.0620, 0.0820, 0.0350, 0.0900, 0.0320, 0.0950, 0.0220, 0.1750, 0, 0.0320, 0.0870, 0, 0.0350, 0, 0, 0.0240, 0.1060])
fbus = np.array([1, 1, 2, 3, 2, 2, 4, 5, 6, 6, 6, 6, 9, 9, 4, 12, 12, 12, 12, 14, 16, 15, 18, 19, 10, 10, 10, 10, 21, 15, 22, 23, 24, 25, 25, 28, 27, 27, 29, 8, 6])
tbus = np.array([2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 10, 11, 10, 12, 13, 14, 15, 16, 15, 17, 18, 19, 20, 20, 17, 21, 22, 22, 23, 24, 24, 25, 26, 27, 27, 29, 30, 30, 28, 28])
rateA = np.array([1.3000, 1.3000, 0.6500, 1.3000, 1.3000, 0.6500, 0.9000, 0.7000, 1.3000, 0.3200, 0.6500, 0.3200, 0.6500, 0.6500, 0.6500, 0.6500, 0.3200, 0.3200, 0.3200, 0.1600, 0.1600, 0.1600, 0.1600, 0.3200, 0.3200, 0.3200, 0.3200, 0.3200, 0.3200,  0.1600, 0.1600, 0.1600, 0.1600, 0.1600,  0.1600, 0.6500, 0.1600, 0.1600, 0.1600, 0.3200,  0.3200])

r_p_jx = np.array([0.0200 + 0.0600*1j, 0.0500 + 0.1900*1j, 0.0600 + 0.1700*1j, 0.0100 + 0.0400*1j, 0.0500 + 0.2000*1j, 0.0600 + 0.1800*1j, 0.0100 + 0.0400*1j, 0.0500 + 0.1200*1j, 0.0300 + 0.0800*1j, 0.0100 + 0.0400*1j, 0.0000 + 0.2100*1j,  0.0000 + 0.5600*1j, 0.0000 + 0.2100*1j,  0.0000 + 0.1100*1j,  0.0000 + 0.2600*1j, 0.0000 + 0.1400*1j, 0.1200 + 0.2600*1j, 0.0700 + 0.1300*1j, 0.0900 + 0.2000*1j, 0.2200 + 0.2000*1j, 0.0800 + 0.1900*1j, 0.1100 + 0.2200*1j,  0.0600 + 0.1300*1j, 0.0300 + 0.0700*1j, 0.0900 + 0.2100*1j, 0.0300 + 0.0800*1j, 0.0300 + 0.0700*1j, 0.0700 + 0.1500*1j, 0.0100 + 0.0200*1j, 0.1000 + 0.2000*1j, 0.1200 + 0.1800*1j, 0.1300 + 0.2700*1j, 0.1900 + 0.3300*1j, 0.2500 + 0.3800*1j, 0.1100 + 0.2100*1j, 0.0000 + 0.4000*1j, 0.2200 + 0.4200*1j, 0.3200 + 0.6000*1j, 0.2400 + 0.4500*1j, 0.0600 + 0.2000*1j, 0.0200 + 0.0600*1j])

gij = np.real(1./r_p_jx)
bij = np.imag(1./r_p_jx)


XY = np.random.random((Num_bus, 2))	# xy coordinates of each node in power grid
G = nx.Graph()
G.add_nodes_from(indices)
for n in range(Num_bus):
	G.add_edge(fbus[n], tbus[n])
#er = nx.erdos_renyi_graph(Num_bus, 0.15)
#G.add_nodes_from(er.nodes)
#G.add_edges_from(er.edges)

#P_D = np.random.random(Num_bus)   # demand power

epsilons = -0.05 + 0.1*np.random.random(Num_bus)
thetas = -np.pi/2 + np.pi*np.random.random(Num_bus)
thetas[0] = 0   	# convention

#a = 0.001*np.random.random(Num_bus)
#b = 0.2*np.random.random(Num_bus)
#c = 0.2*np.random.random(Num_bus)
i = 0
for ed in G.edges:
	G.edges[ed[0], ed[1]]['gij'] = gij[i]
	G.edges[ed[0], ed[1]]['bij'] = bij[i]
	G.edges[ed[0], ed[1]]['rateA'] = rateA[i]
	i += 1

x0 = np.concatenate((thetas, epsilons))
costs = cost_func(x, G, P_D, a, b, c)
print (costs)
eps_min = -0.05
eps_max = 0.05
theta_min = -np.pi/2
theta_max = np.pi/2
#bounds = Bounds([eps_min, eps_max], [theta_min, theta_max])
res = minimize(cost_func, x0, method='TNC', args=(G, P_D, a, b, c, rateA, PMIN, PMAX), options={'verbose': 1})  #, bounds=bounds)
P_t_star = output_P_T(res.x, G, P_D, a, b, c, rateA, PMIN, PMAX)

pdb.set_trace()

print ('******************* OPTIMAL SOLUTION ************')
print ('theta_* = ', res.x[:Num_bus])
print ('epsilon_* = ', res.x[Num_bus:])
print ('*************************************************')
theta_star = res.x[:Num_bus]
epsilon_star = res.x[Num_bus:]
#nx.draw(G, cmap=plt.get_cmap('viridis'), node_color=theta_star, with_labels=True, font_weight='bold', font_color='white')
#plt.show()
##############


Nx = 100
Ny = 100

dx = 1./float(Nx)
dy = 1.0/float(Ny)
cellSize = np.min((dx, dy))
Xc = Nx*dx/2.0
Yc = Ny*dy/2.0

X0u = 0.5 #0.2*Nx*dx
Y0u = 1 #0.1*Ny*dy

X0v = 0.5 #0.2*Nx*dx
Y0v = 0.3 #0.1*Ny*dy

X0w = 0.25 #0.2*Nx*dx
Y0w = 1 #0.1*Ny*dy

# mesh
mesh = Grid2D(dx=dx, dy=dy, nx=Nx, ny=Ny)


def circle(x, y, xc,yc, R):
	arr = []
	for xx, yy in zip(x,y):
		if (xx-xc)**2 + (yy - yc)**2 - R >=0:
			arr.append(0)
		else:
			arr.append(1)
	return np.array(arr)

# IC
x,y = mesh.cellCenters

u = CellVariable(mesh=mesh, name = "u")
arru = circle(x, y, X0u, Y0u, 0.3)
arrv = circle(x, y, X0v, Y0v, 0.3)
arrw = circle(x, y, X0w, Y0w, 0.3)

u.value = arru #(numerix.exp(-50*(x-X0u)**2-50*(y-Y0u)**2))

v = CellVariable(mesh=mesh, name = "v")
v.value = arrv #(numerix.exp(-50*(x-X0v)**2-50*(y-Y0v)**2))

w = CellVariable(mesh=mesh, name = "w")
w.value = arrw #(numerix.exp(-50*(x-X0w)**2-50*(y-Y0w)**2))


# impose Dirichlet BCs
'''
valueBC_u = 0.
u.constrain(valueBC_u, mesh.facesTop)
u.constrain(valueBC_u, mesh.facesBottom)
u.constrain(valueBC_u, mesh.facesRight) 
u.constrain(valueBC_u, mesh.facesLeft)  

valueBC_v = 0.
v.constrain(valueBC_v, mesh.facesTop)
v.constrain(valueBC_v, mesh.facesBottom)
v.constrain(valueBC_v, mesh.facesRight) 
v.constrain(valueBC_v, mesh.facesLeft) 

valueBC_w = 0.
w.constrain(valueBC_w, mesh.facesTop)
w.constrain(valueBC_w, mesh.facesBottom)
w.constrain(valueBC_w, mesh.facesRight) 
w.constrain(valueBC_w, mesh.facesLeft) 
'''


# set coefficients for convection-diffusion equation
diffCoeff = CellVariable(mesh=mesh, rank=1)
diffCoeff.value = D

sourceCoeff = CellVariable(mesh=mesh, rank=0)


timeStepDuration = 0.1 #10 * 0.9 * cellSize**2 / (2 * D)
steps = 9
print (steps)

#viewer = Viewer((u,v))
# solve the eq
x,y = mesh.cellCenters
tt = 0.0
xm = x.reshape((Ny,Nx))
ym = y.reshape((Ny,Nx))
for step in range(steps):
	tt += timeStepDuration
	# set up problem
	# MESS WITH VAULES TO SEE RES FOR SENS
	equ = TransientTerm(var=u) == (ImplicitDiffusionTerm(coeff=diffCoeff, var=u)) + RA*(1 - u - v + w)*u + ksiB*RA*(v-w)*u - u # + ImplicitSourceTerm(coeff=sourceCoeff))
	eqv = TransientTerm(var=v) == (ImplicitDiffusionTerm(coeff=diffCoeff, var=v)) + RB*(1 - u - v + w)*v + ksiA*RB*(u-w)*v - v # + ImplicitSourceTerm(coeff=sourceCoeff))
	eqw = TransientTerm(var=w) == (ImplicitDiffusionTerm(coeff=diffCoeff, var=w)) + ksiA*RB*(u - w)*v + ksiB*RA*(v - w)*u - 2*w # + ImplicitSourceTerm(coeff=sourceCoeff))
	eqn = equ & eqv & eqw
	eqn.solve(dt=timeStepDuration)
	#viewer.plot()
	print ("step ", step, " time= ", tt)

sol_u = u.value
sol_u = sol_u.reshape((Ny,Nx))
sol_v = v.value
sol_v = sol_v.reshape((Ny,Nx))
sol_w = w.value
sol_w = sol_w.reshape((Ny,Nx))
ss = 1 - sol_u - sol_v + sol_w

fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(15, 10))
fig.subplots_adjust(wspace=0.1)
axes[0,0].pcolor(xm, ym, sol_u, cmap='jet', label='u', vmin=0, vmax=1)
##GRAPH U
axes[0,0].set_xlabel('x', fontsize=25)
axes[0,0].set_ylabel('y', fontsize=25)

#GRAPH V
axes[0,1].pcolor(xm, ym, sol_v, cmap='jet', label='v', vmin=0, vmax=1)
axes[0,1].set_xlabel('x', fontsize=25)
axes[0,2].pcolor(xm, ym, sol_w, cmap='jet', label='w', vmin=0, vmax=1)
axes[0,2].set_xlabel('x', fontsize=25)
axes[0,0].legend(fontsize=15)
axes[0,1].legend(fontsize=15)
axes[0,2].legend(fontsize=15)

#CONTROLS GRAPH COLORING OF S
axes[1,0].pcolor(xm, ym, ss, cmap='jet', label='s', vmin=0, vmax=1)
##GRAPH S
axes[1,0].set_xlabel('x', fontsize=25)
axes[1,0].set_ylabel('y', fontsize=25)
axes[1,0].legend(fontsize=15)

nx.draw_networkx(G, ax=axes[1,1], with_labels=False, pos=XY, cmap=plt.get_cmap('jet'), node_color=theta_star,  node_size=100, fontsize=15, font_color='white', label=r'$\theta$')
nx.draw_networkx(G, ax=axes[1,2], with_labels=False, pos=XY, cmap=plt.get_cmap('jet'), node_color=epsilon_star,  node_size=100, fontsize=15, font_color='white', label=r'$\epsilon$')
axes[1,1].legend(fontsize=10, loc=4)
axes[1,2].legend(fontsize=10, loc=4)
axes[1,1].set_xlabel('x', fontsize=25)
axes[1,2].set_xlabel('x', fontsize=25)
plt.tight_layout()
plt.show()
##pdb.set_trace()

