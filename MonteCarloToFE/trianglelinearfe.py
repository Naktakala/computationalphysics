import numpy as np

X = 0
Y = 1

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# General triangle FE view


class TriangleLinearFE:
    # ====================================== Constructor
    def __init__(self, v0, v1, v2):
        self.v0   = np.array(v0)
        self.v1   = np.array(v1)
        self.v2   = np.array(v2)
        self.v01  = self.v1 - self.v0
        self.v02  = self.v2 - self.v0
        self.J    = np.zeros([2, 2])
        self.Jinv = np.zeros([2, 2])
        self.JT = np.zeros([2, 2])
        self.JTinv = np.zeros([2, 2])

        self.compute_jacobian(self.v01,self.v02)

    # ====================================== Jacobian
    def compute_jacobian(self, v01, v02):
        self.J[0, 0] = v01[X]
        self.J[0, 1] = v02[X]
        self.J[1, 0] = v01[Y]
        self.J[1, 1] = v02[Y]

        self.Jinv = np.linalg.inv(self.J)
        self.JT = np.transpose(self.J)
        self.JTinv = np.linalg.inv(self.JT)

        # print("J    : ", self.J)
        # print("Jinv : ", self.Jinv)
        # print("JT   : ", self.JT)
        # print("JTinv: ", self.JTinv)

    # ====================================== Shape function
    def shape_i_xi_eta(self, i, xi, eta):
        if xi > 1.0 or xi < 0.0:
            return 0.0
        if eta > 1.0 or eta < 0.0:
            return 0.0

        if i == 0:
            return 1.0 - xi - eta
        elif i == 1:
            return xi
        elif i == 2:
            return eta

    def shape_i_x_y(self, i, x, y):
        xi_eta = np.matmul(self.Jinv, np.array([x,y])-self.v0)
        xi  = xi_eta[X]
        eta = xi_eta[Y]

        return self.shape_i_xi_eta(i,xi,eta)

    # ====================================== Get integration points
    def get_int_points_and_dxdy(self,res):
        N = res
        dnat = 1.0 / N
        xi = np.zeros(N)
        eta = np.zeros(N)

        x = np.zeros([N * N])
        y = np.zeros([N * N])

        # dd = np.matmul(self.JTinv, np.array([dnat, dnat]))
        dd = np.array([dnat, dnat])*np.linalg.det(self.Jinv)

        print("dd: ",dd)

        k = 0
        for i in range(0, N):
            xi[i] = 0.5 * dnat + i * dnat
            for j in range(0, N):
                eta[j] = 0.5 * dnat + j * dnat
                xy = self.v0 + np.matmul(self.J, np.array([xi[i], eta[j]]))
                x[k] = xy[X]
                y[k] = xy[Y]
                if xi[i] <= (1.00001 - eta[j]):
                    k += 1

        xx = np.zeros([k])
        yy = np.zeros([k])
        for i in range(0, k):
            xx[i] = x[i]
            yy[i] = y[i]

        return xx,yy,dd[0],dd[1]

    # ====================================== Distance to surface
    def distance_to_surface(self,p0,omega):
        p1 = p0 + omega*10.0*np.max(self.J)
        v01 = np.array([self.v01[0], self.v01[1], 0.0])
        v02 = np.array([self.v02[0], self.v02[1], 0.0])
        v12 = v01-v02

        # print("Line ", p0, p1)

        flengths = []
        flengths.append(np.linalg.norm(v01))
        flengths.append(np.linalg.norm(v12))
        flengths.append(np.linalg.norm(v02))

        # print("Lengths ",flengths)

        v01n = v01/np.linalg.norm(v01)
        v12n = v12 / np.linalg.norm(v12)
        v02n = v02/np.linalg.norm(v02)

        flegs = []
        flegs.append(v01n)
        flegs.append(-v12n)
        flegs.append(-v02n)

        fpoints = []
        fpoints.append(np.array([self.v0[0], self.v0[1], 0.0]))
        fpoints.append(np.array([self.v1[0], self.v1[1], 0.0]))
        fpoints.append(np.array([self.v2[0], self.v2[1], 0.0]))

        # print("Points ", fpoints)

        khat = np.array([0.0,0.0,1.0])
        fnorms = []
        for f in range(0,3):
            fnorms.append(np.cross(flegs[f],khat))

        for f in range(0,3):
            vr0 = fpoints[f] - p0
            vr1 = fpoints[f] - p1

            dp0 = np.dot(vr0, fnorms[f])
            dp1 = np.dot(vr1, fnorms[f])

            # print("face ", f, dp0, dp1)

            if (dp0*dp1 < -1.0e-8):
                d2surf = 10.0*(1.0/(1+abs(dp1/dp0)))
                # print(d2surf)
                pc = p0 + d2surf*omega
                d = np.dot(pc-fpoints[f],flegs[f])
                # print(d,flengths[f])

                if (d <= flengths[f] and d >= 0.0):
                    return p0, p0 + d2surf*omega

        return p0, p1
        print("Error no face intersection found")

    def distance_to_surface2(self,p0,p1,omega):
        v01 = np.array([self.v01[0], self.v01[1], 0.0])
        v02 = np.array([self.v02[0], self.v02[1], 0.0])
        v12 = v01-v02

        # print("Line ", p0, p1)
        L = np.linalg.norm(p1-p0)

        flengths = []
        flengths.append(np.linalg.norm(v01))
        flengths.append(np.linalg.norm(v12))
        flengths.append(np.linalg.norm(v02))

        # print("Lengths ",flengths)

        v01n = v01/np.linalg.norm(v01)
        v12n = v12 / np.linalg.norm(v12)
        v02n = v02/np.linalg.norm(v02)

        flegs = []
        flegs.append(v01n)
        flegs.append(-v12n)
        flegs.append(-v02n)

        fpoints = []
        fpoints.append(np.array([self.v0[0], self.v0[1], 0.0]))
        fpoints.append(np.array([self.v1[0], self.v1[1], 0.0]))
        fpoints.append(np.array([self.v2[0], self.v2[1], 0.0]))

        # print("Points ", fpoints)

        khat = np.array([0.0,0.0,1.0])
        fnorms = []
        for f in range(0,3):
            fnorms.append(np.cross(flegs[f],khat))

        points = []
        lengths = []
        for f in range(0,3):
            vr0 = fpoints[f] - p0
            vr1 = fpoints[f] - p1

            dp0 = np.dot(vr0, fnorms[f])
            dp1 = np.dot(vr1, fnorms[f])

            # print("face ", f, dp0, dp1)

            if (dp0*dp1 < -1.0e-8):
                d2surf = L*(1.0/(1+abs(dp1/dp0)))
                # print(d2surf)
                pc = p0 + d2surf*omega
                d = np.dot(pc-fpoints[f],flegs[f])
                # print(d,flengths[f])

                if (d <= flengths[f] and d >= 0.0):
                    points.append(p0 + d2surf*omega)
                    lengths.append(d2surf)

        if len(points) == 1:
            return p0,points[0],True
        elif len(points) == 2:
            return points[0],points[1],True

        return p0, p1, False
        print("Error no face intersection found")