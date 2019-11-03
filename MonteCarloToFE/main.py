import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import trianglelinearfe as tfe
import math


class Line:
    def __init__(self,v0,v1):
        if np.size(v0) == 2:
            self.v0 = np.array([v0[0],v0[1],0.0])
        else:
            self.v0 = np.array(v0)

        if np.size(v1) == 2:
            self.v1 = np.array([v1[0],v1[1],0.0])
        else:
            self.v1 = v1
        self.v01 = self.v1 - self.v0
        self.length = np.linalg.norm(self.v01)
        self.omega = self.v01/self.length
        self.x = [self.v0[0], self.v1[0]]
        self.y = [self.v0[1], self.v1[1]]

def limit(mini,maxi,value):
    if value < mini:
        return mini
    if value > maxi:
        return maxi

    return value


print("Hello world")
tri = tfe.TriangleLinearFE([0.25, 0.25], [1.5, 0.5], [0.6, 1.6])
# tri = tfe.TriangleLinearFE([1.00, 0.00], [2.0, 1.0], [1, 2.0])

x,y,dx,dy = tri.get_int_points_and_dxdy(300)
xmin = np.min(x)
xmax = np.max(x)
ymin = np.min(y)
ymax = np.max(y)
N = np.size(x)
v = np.zeros([N])
vi = np.zeros([3,N])

node_V = np.array([0.0, 0.0, 0.0])
for i in range(0,N):
    v[i] = tri.shape_i_x_y(0,x[i],y[i])
    for w in range(0,3):
        node_V[w] += tri.shape_i_x_y(w,x[i],y[i]) * np.linalg.det(tri.J)/300/300
        vi[w,i] = tri.shape_i_x_y(w,x[i],y[i])


total_V = np.sum(node_V)
print("Total integral: ",total_V,node_V)

# ========================================== Create lines
np.random.seed(0)
N_l = 80
dl = 2.0/N_l
lines = []
Doff = 20
for i in range(0, N_l):
    # xc = 0.5*dl + i*dl
    xc = 0.0 + np.random.random()*2.0
    yc = 0.0+ np.random.random()*1.0
    # theta = 2.0*math.pi*np.random.random()
    theta = 0.0
    x0 = xc - 10.0*math.cos(theta)
    y0 = yc - 10.0*math.sin(theta)
    x1 = xc + 10.0 * math.cos(theta)
    y1 = yc + 10.0 * math.sin(theta)

    # x0 = limit(0.0,2.0,x0)
    # y0 = limit(0.0,2.0,y0)
    # x1 = limit(0.0,2.0,x1)
    # y1 = limit(0.0,2.0,y1)

    lines.append(Line([x0,y0],[x1,y1]))
    # print(x0,y0,x1,y1)



# v0,v1 = tri.distance_to_surface(lines[0].v0, lines[0].omega)

is_lines = []
for i in range(0, N_l):
    v0,v1, intersects = tri.distance_to_surface2(lines[i].v0, lines[i].v1,lines[i].omega)
    if (intersects):
        is_lines.append(Line(v0, v1))

# ================================ Compute weighted track lengths
Ns = 20
phi = np.array([0.0, 0.0, 0.0])
total_tl = 0.0
N_il = len(is_lines)
for i in range(0, N_il):
    line = is_lines[i]
    L  = line.length
    dL = L/Ns
    weights = np.array([0.0,0.0,0.0])
    for s in range(0,Ns):
        p = line.v0 + 0.5*line.omega*dL + s*line.omega*dL
        for w in range(0,3):
            weights[w] += tri.shape_i_x_y(w, p[0], p[1])*dL

    phi += weights/node_V
    total_tl += L


print("Nodal phi      : ", phi/N_l)
print("Node avg       : ", np.average(phi/N_l))
print("Avg tracklength: ", total_tl/N_l)
print("Avg flux       : ", total_tl/N_l/total_V)

# ========================================== Plotting nodal values
plt.figure()

v = np.zeros([N])
for i in range(0,N):
    for w in range(0,3):
        v[i] += tri.shape_i_x_y(w,x[i],y[i]) * phi[w]
plt.scatter(x, y, c=v, cmap=cm.jet,vmin=0.0, s=0.1)

# ========================================== Plotting shape functions
fig, axs = plt.subplots(1,3,figsize=(7,2))

levels = np.linspace(0.0,1.0,124)
for w in range(0,3):
    axs[w].scatter(x, y, c=vi[w,:], cmap=cm.jet,vmin=0.0, s=0.1)
    axs[w].set_xlim(0, 1.75)
    axs[w].set_ylim(0, 1.75)

# ========================================== Plot the tracelines
# plt.figure()
# for i in range(0, N_l):
#     plt.plot(lines[i].x,lines[i].y,'k-',linewidth=1)
#
# for i in range(0, N_il):
#     plt.plot(is_lines[i].x,is_lines[i].y,'ko',markersize=3,linewidth=1)

plt.xlim(0,1.75)
plt.ylim(0,1.75)
plt.show()


