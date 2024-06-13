import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt

from Library.utilities import *


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
        "init_func": "jump1",
        "tf": 0.1,
        "Theta_st":.5,
        "Theta_min":0.2,
        "Theta_max":1,
        "Scheme": "SATh", #"Theta"
        "Theta_computation":"Fixed-Point", #"Newton"
        "Theta_choice_epsilon":1e-6,
        "Method_convergence_epsilon":1e-6,
        "Timer":False,
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

        if self.default_dict["Scheme"] == "Theta":  #Simple/Standard Theta Scheme
            self.theta = self.default_dict["Theta_st"]
            self.sol_num = self.Theta_Scheme()
            self.sol_exc = self.sol_ex()
        
        elif self.default_dict["Scheme"] == "SATh":  #Self-Adaptive Theta Scheme
            self.solver = SATh_Solver(self)
            self.sol_num = self.solver.SATh_Scheme()
            self.sol_exc = self.sol_ex()

        else:
            print("Wrong Scheme type in the inputs")
            
    def print_params(self):
        return self.default_dict

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

        self.theta_computation = self.env.default_dict["Theta_computation"]

        self.lam = env.mesh.dt/env.mesh.dx #CFL, here with the case of the flux function f(u) = au
                                           #where a=1
        #self.cyclicpermut_mat = sparse.csr_matrix(np.roll(np.eye(env.mesh.Nx), shift=-1, axis=0))
        #self.d = lambda u: self.env.funcs.arrange_u(u,-1,bc=self.default_dict["Boundary"])

    def interpol_quadrature(self):  #use with tau=1/2 for standard linear interpolation
        #return ((self.env.mesh.dt/2)*self.u_down + (1 - self.env.mesh.dt/2)*self.u_up)/self.env.mesh.dt
        return self.u_down/2 + self.u_up/2
    
    def timer(self, keyword=None):
        if keyword == "set":
            print(int(self.env.tf%self.env.mesh.dt) * "-")
        else:
            print("-",end="")
    
    def theta_choice(self, v, w, epsilon=1e-6):
        if np.abs(w) > epsilon:
            #print(np.abs(v/w > 1))
            #if np.abs(v/w) > self.theta_min:
            #    print("theta=v/w",self.env.default_dict["Theta_computation"])
            #    print(max(self.theta_min, np.abs(v/w) ))
            return min(max(self.theta_min, np.abs(v/w) ), 1)  #
        else:
            return self.theta_st
    

    def F(self, w, v, i):
        return np.array([
            w[i] + self.lam * self.thetas[i] * w[i] + self.lam * (self.u_up[i] - self.thetas[i-1]*w[i-1] - self.u_up[i-1]),
            v[i] + self.lam * .5 * self.thetas[i]**2 * w[i] + self.lam * .5 * (self.u_up[i] - self.thetas[i-1]**2 * w[i-1] - self.u_up[i-1])
        ])

    def compute_J(self, w, v, i, epsilon=1e-6):    #Compute the Jacobian
        if np.abs(w[i])>epsilon :
            if v[i]/w[i] > self.theta_min:
                J = np.array([[1, self.lam], [-(v[i]**2)*self.lam / (2*(w[i]**2)), 1 + self.lam/w[i]]])
            else :
                J = np.array([[1+self.lam*self.theta_min, 0], [self.lam * (self.theta_min**2) / 2, 1]])
        else :
            J = np.array([[1+self.lam*self.theta_st, 0], [self.lam * (self.theta_st**2) / 2, 1]])
        return J


    def Newton(self, epsilon=1e-6):
        w_ = np.copy(self.u_down)
        w_[0] = 0
        #v = np.empty_like(self.u_down)
        v = self.u_til - self.u_down
        v[0] = 0 #
        w = np.zeros_like(w_)
        #(Neumann at the left)

        while np.linalg.norm(w - w_) > epsilon:
            for i in range(1,self.env.mesh.Nx +1):
                w_[i] = w[i]

                self.thetas[i] = self.theta_choice(v[i],w[i],epsilon=self.env.default_dict["Theta_choice_epsilon"])

                #compute step (linear system resolution)
                s = np.linalg.solve(self.compute_J(w,v,i, epsilon=self.env.default_dict["Theta_choice_epsilon"]), -self.F(w,v,i))
                #iterate Newton problem
                w[i] += s[0]
                v[i] += s[1]
        
        return [w,v]


    def fixed_point_iter(self, theta_left, u_down_left, u_down,
                                      w_left, lam, epsilon=1e-6):
        w = w_left #
        w_ = epsilon + w  #Just to initialize
        theta = self.theta_st   #Try wit hthe previous time step theta

        while np.abs(w_ - w) >= epsilon:
            w_ = w
            w = (u_down - theta_left*w_left - (u_down_left + w_left)) / (1 + lam*theta)
            v = (-lam/2) * (u_down - theta_left**2 * w_left - (u_down_left + w_left) + theta**2*w)
            theta = self.theta_choice(v,w, epsilon=self.env.default_dict["Theta_choice_epsilon"])
        
        return theta, w
        

    def update_thetas(self):

        if self.theta_computation == "Fixed-Point":
            #self.u_down[0] = self.left_bound_val
            #w_left = self.u_up[0] - self.u_down[0]  #Boundary condition: the value at the left is constant
            w_left = 0  #For now, we work with a constant flux at the left, so the first w_left is equal to 0

            for i in range(1, self.env.mesh.Nx +1):
                theta, w_left = self.fixed_point_iter(self.thetas[i-1], self.u_down[i-1],
                                    self.u_down[i], w_left, self.lam, epsilon=self.env.default_dict["Method_convergence_epsilon"])
                self.thetas[i] = theta

        elif self.theta_computation == "Newton":
            [w,v] = self.Newton(epsilon=self.env.default_dict["Method_convergence_epsilon"])
            for i in range(w.size):
                self.thetas[i] = self.theta_choice(v[i],w[i], epsilon=self.env.default_dict["Theta_choice_epsilon"])

            self.u_til = v


    def SATh_Scheme(self):
        t = 0
        self.thetas = np.ones(self.env.mesh.Nx+1) * self.theta_st
        self.u_down = self.env.funcs.init_sol.copy()
        self.u_up = np.empty_like(self.u_down)
        self.u_up[0] = self.u_down[0]

        if self.env.default_dict["Timer"]==True:
            self.timer("set")

        while (t<self.env.tf):

            if t!=0:
                self.update_thetas()
            #print(min(self.thetas), max(self.thetas))
            
            coefs = self.env.mesh.dt*(np.eye(self.env.mesh.Nx+1)-np.diag(self.thetas))
            A = self.env.mats.Iter_Mat(self.env.mesh, self.thetas, self.env.alpha, adaptive=True)
            #A = sparse.linalg.LinearOperator((self.env.mesh.Nx+1,self.env.mesh.Nx+1), matvec=self.env.funcs.Iter_Func)
            b = self.u_down - self.env.alpha * coefs@(self.env.mats.Dx @ self.u_down)
            b[0] = self.u_down[0]

            self.u_up, _ = gmres(A, b)
            #self.u_up = A.matvec(b)
            self.u_down = self.u_up.copy()
            self.u_til += self.u_up    #Only needed in the Newton method to update v (-> see line 176)

            t += self.env.mesh.dt
            if self.env.default_dict["Timer"]==True:
                self.timer()
                #print(t)

        return self.u_up

