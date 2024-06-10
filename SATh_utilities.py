import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt


class Problem():
    def __init__(self, **kwargs):

        self.default_dict = {
        "a": 0,
        "b": 1,
        "Nx": 500,
        "cfl": 5,
        "alpha": 1,
        "mesh_type": "offset_constantDx",
        "params": 0.2,
        "init_func": "bell",
        "tf": 0.1,
        "Theta_st":.5,
        "Theta_min":0.2,
        "Theta_max":1,
        "Scheme": "Theta",
        "Path": "pictures",
        "Boundary":"constant"
        }
        #replacing by input values in the dict:
        for key in kwargs.keys():
            self.default_dict[key] = kwargs[key]

        self.params = self.default_dict["params"]
        self.tf = self.default_dict["tf"]
        self.alpha = self.default_dict["alpha"]

        self.mesh = Mesh(self.default_dict["a"], self.default_dict["b"], 
                         self.default_dict["Nx"], self.default_dict["cfl"], 
                         self.default_dict["Boundary"], self.default_dict["mesh_type"])
        self.mats = Matrices(self.mesh, self.default_dict["Boundary"], self.alpha)
        self.funcs = Functions(self.mesh, self.params, self.tf, self.default_dict["init_func"])

        if self.default_dict["Scheme"] == "Theta":
            self.theta = self.default_dict["Theta_st"]
            self.sol_num = self.Theta_Scheme()
            self.sol_exc = self.sol_ex()
        
        else: #Self-Adaptive Theta Scheme
            self.solver = SATh_Solver(self)
            self.sol_num = self.solver.SATh_Scheme()
            self.sol_exc = self.sol_ex()
            
            
    def sol_ex(self):
        return self.funcs.exact(self.mesh.nodes, self.params, self.tf)

    def Theta_Scheme(self):
        t = 0
        u = self.funcs.init_sol.copy()
        coef = self.mesh.dt*(1-self.theta)
        A = self.mats.Iter_Mat(self.mesh, self.theta, self.alpha, adaptive=False)
        
        while (t<self.tf):
            t += self.mesh.dt
            b = u - coef*self.alpha*(self.mats.Dx @ u)
            u, _ = gmres(A, b)

        return u


class SATh_Solver:     #Works only for nonperiodical boundary conditions!
    def __init__(self, env):
        self.env = env
        self.theta_st = env.default_dict["Theta_st"]
        self.theta_min = env.default_dict["Theta_min"]
        self.thetas = np.ones(env.mesh.Nx+1) * self.theta_st

        self.u_down = self.env.funcs.init_sol.copy()
        self.u_til = np.empty_like(self.u_down)
        self.u_up = np.empty_like(self.u_down)
        self.left_bound_val = self.env.funcs.init_sol[0]

        self.lam = env.mesh.dt/env.mesh.dx #CFL, here with the case of the flux function f(u) = au
                                           #where a=1
        #self.cyclicpermut_mat = sparse.csr_matrix(np.roll(np.eye(env.mesh.Nx), shift=-1, axis=0))
        #self.d = lambda u: self.env.funcs.arrange_u(u,-1,bc=self.default_dict["Boundary"])

    def interpol_quadrature(self):  #use with tau=1/2 for standard linear interpolation
        #return ((self.env.mesh.dt/2)*self.u_down + (1 - self.env.mesh.dt/2)*self.u_up)/self.env.mesh.dt
        return self.u_down/2 + self.u_up/2
    
    
    def theta_choice(self, v, w, epsilon=1e-6):
        if np.abs(w) > epsilon:
            return min(max(self.theta_min, np.abs(v/w) ),1 )
        else:
            return self.theta_st
    

    def F(self, w, v, u, lam, thetas, i):
        return np.array([
            w[i] + lam * thetas[i] * w[i] + lam * (u[i] - thetas[i-1]*w[i-1] - u[i-1]),
            v[i] + lam * .5 * thetas[i]**2 * w[i] + lam * .5 * (u[i] - thetas[i-1]**2 * w[i-1] - u[i-1])
        ])

    def compute_J(self, w, v, u, lam, thetas, i, epsilon=1e-6):    #Compute the Jacobian
        if np.abs(w[i])>epsilon :
            if v[i]/w[i] > self.theta_min:
                J = np.array([[1, lam], [-v[i]**2*lam / (2*w[i]**2), 1 + lam/w[i]]])
            else :
                J = np.array([[1+lam*self.theta_min, 0], [lam * self.theta_min**2 / 2, 1]])
        else :
            J = np.array([[1+lam*self.theta_st, 0], [lam * self.theta_st**2 / 2, 1]])
        return J


    def Newton(self, u, epsilon=1e-6):
        u_next = np.zeros_like(u)

        while np.linalg.norm(u_next-u):
            self.compute_J()


        pass

    def fixed_point_iter(self, theta_left, u_down_left, u_down,
                                      w_left, lam, epsilon=1e-6):
        w = w_left #
        w_ = epsilon + w  #To initialize
        theta = self.theta_st

        while np.abs(w_ - w) > epsilon:
            w_ = w
            w = (u_down - theta_left*w_left - (u_down_left + w_left)) / (1 + lam*theta)
            v = (-lam/2) * (u_down - theta_left**2 * w_left - (u_down_left + w_left) + theta**2*w)
            theta = self.theta_choice(v,w)
        
        return theta, w
        

    def update_thetas(self):
        #self.u_down[0] = self.left_bound_val
        #w_left = self.u_up[0] - self.u_down[0]  #Boundary condition: the value at the left is constant
        w_left = 0  #For now, we work with a constant flux at the left, so the first w_left is equal to 0

        for i in range(1, self.env.mesh.Nx +1):
            theta, w_left = self.fixed_point_iter(self.thetas[i-1], self.u_down[i-1],
                                  self.u_down[i], w_left, self.lam)
            self.thetas[i] = theta


    def SATh_Scheme(self):
        t = 0
        self.thetas = np.ones(self.env.mesh.Nx+1) * self.theta_st
        self.u_down = self.env.funcs.init_sol.copy()
        self.u_up = np.empty_like(self.u_down)
        self.u_up[0] = self.u_down[0]

        """
        while (t<self.env.tf):
            w_left = self.u_up[0] - self.u_down[0]
            
            for i in range(1, self.env.mesh.Nx +1):
                self.thetas[i], w_left = self.fixed_point_iter(self.thetas[i-1], self.u_up[i-1],
                                  self.u_down[i], w_left, self.lam)

                self.u_up[i] = (self.u_down[i] + self.env.alpha * self.env.mesh.dt * (1-self.thetas[i]) * (self.u_down[i]-self.u_down[i-1])/self.env.mesh.dx
                                + self.env.alpha * self.thetas[i] * self.env.mesh.dt * self.u_up[i-1] / self.env.mesh.dx) / (1 + self.env.alpha*self.thetas[i]*self.env.mesh.dt/self.env.mesh.dx)
        """

        self.thetas = np.ones(self.env.mesh.Nx+1) * self.theta_st
        self.u_down = self.env.funcs.init_sol.copy()
        self.u_up = np.empty_like(self.u_down)
        self.u_up[0] = self.u_down[0]
        coefs = self.env.mesh.dt*(np.eye(self.env.mesh.Nx+1)-np.diag(self.thetas))
        A = self.env.mats.Iter_Mat(self.env.mesh, self.thetas, self.env.alpha, adaptive=True)
        b = self.u_down - self.env.alpha * coefs@(self.env.mats.Dx @ self.u_down)
        b[0] = self.u_down[0]

        while (t<self.env.tf):
            self.u_up, _ = gmres(A, b)
            self.u_down = self.u_up.copy()

            self.update_thetas()
            
            coefs = self.env.mesh.dt*(np.eye(self.env.mesh.Nx+1)-np.diag(self.thetas))
            A = self.env.mats.Iter_Mat(self.env.mesh, self.thetas, self.env.alpha, adaptive=True)
            b = self.u_down - self.env.alpha * coefs@(self.env.mats.Dx @ self.u_down)
            b[0] = self.u_down[0]

            t += self.env.mesh.dt

        return self.u_up


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
            A = Id + self.Dx * theta * mesh.dt * alpha
            line = np.zeros(A.shape[1])
            line[0]=1
            A[0] = line
            return sparse.csr_matrix(A)

        elif adaptive==True:
            if theta.size != mesh.Nx+1:
                print("parameter theta does not have a correct size")
            Id = np.identity(mesh.Nx+1)
            thetas = np.diag(theta) - np.diag(theta[1:],k=-1)
            A = Id + (self.Dx + thetas) * mesh.dt * alpha
            line = np.zeros(A.shape[1])
            line[0]=1
            A[0] = line
            return sparse.csr_matrix(A)


class Functions():
    def __init__(self, mesh, params, tf, init_type):
        if init_type=="bell":
            self.init_func = self.init_bell
        elif init_type=="jump_nonperiodical":
            self.init_func = self.init_jump1
        elif init_type=="jump_periodical":
            self.init_func = self.init_jump2

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
    
    def arrange_u(self, u, offset, bc):    #This function prepares the values of an iterated solution at a certain step
                                               #in need of a shift in space coefficients.
        ret = np.empty_like(u)
        n = u.size
        
        if bc == "periodical":
            ret[offset%n:] = u[:-offset]
            ret[:offset%n] = u[-offset:]

        elif bc == "none":
            pass
        else :
            print("Invalid boundary type")

        return ret
    