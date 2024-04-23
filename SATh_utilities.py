import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt


class Problem():
    def __init__(self, **kwargs):

        self.params = kwargs["params"]
        self.tf = kwargs["tf"]

        self.mesh = Mesh(kwargs["a"], kwargs["b"], 
                         kwargs["Nx"], kwargs["cfl"], 
                         kwargs["mesh_type"])
        self.mats = Matrices(self.mesh)
        self.funcs = Functions(self.mesh, self.params, self.tf, kwargs["init_func"])

        if kwargs["Scheme"] == "Theta":
            self.theta = kwargs["theta"]
            self.sol_num = self.Theta_Scheme()
            
    def sol_ex(self):
        return self.funcs.exact(self.mesh.nodes, self.params, self.tf)

    def Theta_Scheme(self):
        t = 0
        u = self.funcs.init_sol.copy()
        coef = self.mesh.dt*(1-self.theta)
        A = self.mats.Iter_Mat(self.mesh, self.theta, adaptive=False)
        
        while (t<self.tf):
            t += self.mesh.dt
            b = u - coef*(self.mats.Dx @ u)
            u, _ = gmres(A, b)

        return u

    def SATh_Scheme(self, **kwargs):
        pass


class Mesh:
    def __init__(self, a, b, Nx, cfl, mesh_type="offset_constantDx"):
        self.a = a
        self.b = b
        self.Nx = Nx
        self.cfl = cfl
        self.type = mesh_type

        if self.type == "offset_constantDx":
            self.dx = (self.b - self.a)/(self.Nx)
            self.dt = self.cfl*self.dx
            self.nodes, self.interfaces = self.grid_offset()

        else:
            print("Wrong mesh type")
    
    def set_dt(self, dt):
        self.dt = dt

    def grid_offset(self):
        x = []
        inter = []
        for j in range(self.Nx+1):
            x.append((j)*self.dx -self.a)
            if j != self.Nx:
                inter.append(x[j] + self.dx)
        return np.array(x), np.array(inter)


class Matrices():
    #The matrices depend of the parameters 'boundary' and 'direction' -> boundary conditions and space scheme
    def __init__(self, mesh, boundary="periodical", direction="toRight"):

        if boundary == "periodical" and direction == "toRight":
            self.Dx = self.Dx_PtR(mesh)

    def Dx_PtR(self, mesh):
        ret = sparse.diags([np.ones(mesh.Nx+1),-np.ones(mesh.Nx)],
                           [0,-1], shape=(mesh.Nx+1,mesh.Nx+1), format="lil")
        ret[0,-1] = 1
        return ret/mesh.dx
    
    def Iter_Mat(self, mesh, theta, adaptive):
        Id = sparse.identity(mesh.Nx+1, format="lil")

        if adaptive==False:
            return Id + theta * mesh.dt * self.Dx

        elif adaptive==True:
            if theta.size != mesh.Nx+1:
                print("parameter theta does not have a correct size")
            thetas = sparse.diags(theta, shape=(mesh.Nx+1,mesh.Nx+1))
            return Id + theta + mesh.dt * self.Dx


class Functions():
    def __init__(self, mesh, params, tf, init_type="bell"):
        if init_type=="bell":
            self.init_func = self.init_bell
        elif init_type=="jump_nonperiodical":
            self.init_func = self.init_jump1
        elif init_type=="jump_periodical":
            self.init_func = self.init_jump2

        self.init_sol = self.init_func(mesh.nodes, params)
        self.exact_sol = self.exact(mesh.nodes, params, tf)

    def init_bell(self, x, param, sigma=0.05): #To make a kind of bell curve -> continuous distribution centered in d0=param
        return np.exp(-0.5*((x-param)**2)/sigma**2)

    def init_jump1(self, x, param): #To make a piecewise-constant function with a discontinuity in d0=param (1 before, 0 after)
                                 #not compatible with periodical boundaries, shape: ___
                                 #                                                     |___
        u = np.zeros_like(x, dtype=float)
        for i in range(u.shape[0]):
            if (x[i]<param):
                u[i] = 1
        return u
    
    def init_jump2(self, x, params):  #This one is compatible with periodical boundaries
                                      #shape:     ___
                                      #       ___|   |___
        u = np.zeros_like(x, dtype=float)
        for i in range(u.shape[0]):
            if (x[i]<params[1] and x[i]>=params[0]):
                u[i] = 1
        return u

    def exact(self, x, params, tf):
        tf = np.ones_like(x)*tf
        x0 = x-tf
        u0 = self.init_func(x0, params)
        return u0
    