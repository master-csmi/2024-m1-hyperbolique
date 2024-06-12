import numpy as np
import scipy.sparse as sparse


class Mesh:
    def __init__(self, a, b, Nx, cfl, boundary, mesh_type="offset_constantDx"):
        self.a = a
        self.b = b
        self.Nx = Nx
        if boundary == "constant":  #Neumann at the left
            self.Nx += 1
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
    def __init__(self, mesh, boundary, alpha):

        if alpha >=0 :
            self.Dx = self.Dx_PtR(mesh, boundary)
        else:
            print("alpha<0 not available yet")

    def Dx_PtR(self, mesh, b):
        if b == "periodical":
            ret = sparse.diags([np.ones(mesh.Nx+1),-np.ones(mesh.Nx)],
                            [0,-1], shape=(mesh.Nx+1,mesh.Nx+1), format="lil")
            ret[0,-1] = -1
            
        elif b == "constant":
            dia = np.ones(mesh.Nx+1)
            dia[0] = 0
            ret = sparse.diags([dia,-np.ones(mesh.Nx)],
                            [0,-1], shape=(mesh.Nx+1,mesh.Nx+1), format="lil")
        
        return sparse.csr_matrix(ret/mesh.dx)
    
    def Iter_Mat(self, mesh, theta, alpha, adaptive):

        if adaptive==False:
            Id = sparse.identity(mesh.Nx+1, format="lil")
            Dx = self.Dx.tolil()
            A = Id + Dx * theta * mesh.dt * alpha
            A[0,0] = 1
            return sparse.csr_matrix(A)

        elif adaptive==True:
            if theta.size != mesh.Nx+1:
                print("parameter theta does not have a correct size")
            Id = np.identity(mesh.Nx+1)
            Dx = self.Dx.toarray()
            A = Id + ((Dx * theta[:,np.newaxis]) * mesh.dt * alpha)
            A[0,0] = 1
            return sparse.csr_matrix(A)


class Functions():
    def __init__(self, mesh, params, tf, init_type):
        if init_type=="bell":
            self.init_func = self.init_bell
        elif init_type=="jump1":
            self.init_func = self.init_jump1
        elif init_type=="jump2":
            self.init_func = self.init_jump2
        else:
            print("invalid init function type")

        self.init_sol = self.init_func(mesh.nodes, params)
        self.exact_sol = self.exact(mesh.nodes, params, tf)

    def init_bell(self, x, param, sigma=0.05): #To make a kind of bell curve -> continuous distribution centered in d0=param
        return np.exp(-0.5*((x-param[0])**2)/sigma**2)

    def init_jump1(self, x, param): #To make a piecewise-constant function with a discontinuity in d0=param (1 before, 0 after)
                                 #not compatible with periodical boundaries, shape: ___
                                 #                                                     |___
        u = np.zeros_like(x, dtype=float)
        for i in range(u.shape[0]):
            if (x[i]<param[0]):
                u[i] = 1
        return u
    
    def init_jump2(self, x, params):  #
                                      #shape:     ___
                                      #       ___|   |___
        if len(params)!=2:
            print("2 values needed for the coordinates of the perturbation")
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

    