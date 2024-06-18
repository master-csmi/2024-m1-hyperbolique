import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt

from Library.utilities import *


class Problem():
    def __init__(self, **kwargs):

        self.params_dict = {
        "a": 0,
        "b": 1,
        "Nx": 500,
        "cfl": 5,
        "f": "linear_advection",
        "alpha": 1,
        "mesh_type": "offset_constantDx",
        "params": 0.2,
        "init_func": "jump1",
        "tf": 0.1,
        "Theta_st":.5,
        "Theta_min":0.2,
        "Theta_max":1,
        "Scheme": "SATh", #/"Theta"
        "Flux": "UP", #/"LF"
        "Theta_computation":"Newton", #"Fixed-Point" has been abandoned
        "Theta_choice_epsilon":1e-6,
        "Theta_choice_method":"MinMax", #/"Smoothing"
        "Method_convergence_epsilon":1e-6,
        "Timer":False,
        "print_Newton_iter":False,
        "Path": "pictures",
        "Boundary":"constant",
        "tanh_factor":100,
        }
        #replacing by input values in the dict:
        for key in kwargs.keys():
            self.params_dict[key] = kwargs[key]

        self.params = self.params_dict["params"]
        self.tf = self.params_dict["tf"]
        self.alpha = self.params_dict["alpha"]

        self.mesh = Mesh(self.params_dict["a"], self.params_dict["b"], 
                         self.params_dict["Nx"], self.params_dict["cfl"], 
                         self.params_dict["Boundary"], self.params_dict["mesh_type"])
        self.mats = Matrices(self.mesh, self.params_dict["Boundary"], self.alpha)
        self.funcs = Functions(self.mesh, self.params, self.tf, self.params_dict["init_func"])

        if self.params_dict["Scheme"] == "Theta":  #Simple/Standard Theta Scheme
            self.theta = self.params_dict["Theta_st"]
            scheme = Theta_Scheme(self)
            self.sol_num = scheme.solver()
            self.sol_exc = self.sol_ex()
        
        elif self.params_dict["Scheme"] == "SATh":  #Self-Adaptive Theta Scheme
            self.solver = SATh_Solver(self)
            self.sol_num = self.solver.SATh_Scheme()
            self.sol_exc = self.sol_ex()

        else:
            print("Wrong Scheme type in the inputs")
            
    def print_params(self):
        return self.params_dict

    def sol_ex(self):
        return self.funcs.exact(self.mesh.nodes, self.params, self.tf)
    
    def f(self, u):
        if self.params_dict["f"] == "linear_advection":
            return self.alpha * u



class SATh_Solver:     #Works only for nonperiodical boundary conditions!
    def __init__(self, env):
        self.env = env
        self.theta_st = env.params_dict["Theta_st"]
        self.theta_min = env.params_dict["Theta_min"]
        self.thetas = np.ones(env.mesh.Nx+1) * self.theta_st

        self.u_down = self.env.funcs.init_sol.copy()
        self.u_up, self.u_til = np.empty_like(self.u_down), np.empty_like(self.u_down)
        self.left_bound_val = self.env.funcs.init_sol[0]   #!Change to self.bound_vals = [left,right]

        self.w, self.v = np.empty_like(self.u_down), np.empty_like(self.u_down)

        self.theta_computation = self.env.params_dict["Theta_computation"]

        if env.params_dict["Flux"]=="UP":
            self.alpha = self.env.alpha
            self.lam = env.mesh.dt/env.mesh.dx #CFL, here with the case of the flux function f(u) = au
        elif env.params_dict["Flux"]=="LF":
            self.alpha = self.alpha_LF()
            self.lam = .5 * env.mesh.dt/env.mesh.dx
        else: print("Unknown Flux type")
                                           

        self.tanh_factor = self.env.params_dict["tanh_factor"]

    def timer(self, action=None):
        if action == "set":
            print(int(self.env.tf//self.env.mesh.dt) * "_")
        else:
            print("-",end="")

    def sech(self, x):
        return 1/np.cosh(x)
    
    def alpha_LF(self):
        pass

    def theta_choice(self, i, epsilon=1e-6):
        if self.env.params_dict["Theta_choice_method"] == "MinMax":
            if np.abs(self.w[i]) > epsilon:
                self.thetas[i] = min(max(self.theta_min, np.abs(self.v[i]/self.w[i]) ), 1)
            else:
                self.thetas[i] = self.theta_st
        
        elif self.env.params_dict["Theta_choice_method"] == "Smoothing":
            if self.w[i]==0:
                self.thetas[i] = 1
            else:
                self.thetas[i] = .75 + .25 * np.tanh(self.tanh_factor * (-.5 + self.v[i]/self.w[i]))
        
        else :
            print("Wrong Theta choice method type")
    

    def F(self, i):
        if self.env.params_dict["Flux"]=="UP":
            return np.array([
                self.w[i] + self.lam * self.thetas[i] * self.w[i] + self.lam * (self.env.f(self.u_up[i]) - self.thetas[i-1]*self.w[i-1] - self.env.f(self.u_up[i-1])),
                self.v[i] + self.lam * .5 * self.thetas[i]**2 * self.w[i] + self.lam * .5 * (self.env.f(self.u_up[i]) - self.thetas[i-1]**2 * self.w[i-1] - self.env.f(self.u_up[i-1]))
            ])
        
        elif self.env.params_dict["Flux"]=="LF":
            return np.array([
                self.w[i] + self.lam * (self.thetas[i+1] * (self.f(self.w[i+1] + self.u_up[i+1]) - self.f(i+1) - self.alpha*self.w[i+1])
                                        + 2*self.alpha*self.thetas[i]*self.w[i] - self.thetas[i-1] * (self.f(self.w[i-1] + self.u_up[i-1]) - self.f(i-1) + self.alpha*self.w[i-1]))
                    + self.lam * ((self.f(i+1) - self.f(i-1)) - self.alpha*(self.u_up[i+1] - 2*self.u_up[i] + self.u_up[i-1])),

                self.v[i] + self.lam * ((.5*self.thetas[i+1]**2) * (self.f(self.w[i+1] + self.u_up[i+1]) - self.f(i+1) - self.alpha*self.w[i+1])
                                        + 2*self.alpha*(.5*self.thetas[i]**2)*self.w[i] - (.5*self.thetas[i-1]**2) * (self.f(self.w[i-1] + self.u_up[i-1]) - self.f(i-1) + self.alpha*self.w[i-1]))
                    + .5*self.lam * ((self.f(i+1) - self.f(i-1)) - self.alpha*(self.u_up[i+1] - 2*self.u_up[i] + self.u_up[i-1]))
            ])

    def compute_J(self, i, epsilon=1e-6):    #Compute the Jacobian
        if self.env.params_dict["Theta_choice_method"] == "MinMax":
            if self.env.params_dict["Flux"]=="UP":
                if self.env.params_dict["f"] == "linear_advection":
                    if np.abs(self.w[i])>epsilon :
                        if self.v[i]/self.w[i] > self.theta_min:
                            J = np.array([[1, self.alpha * self.lam], [-(self.v[i]**2)*self.alpha*self.lam / (2*(self.w[i]**2)), 1 + self.alpha*self.lam/self.w[i]]])
                        else :
                            J = np.array([[1+self.lam*self.theta_min*self.alpha, 0], [self.lam * self.alpha*(self.theta_min**2) / 2, 1]])
                    else :
                        J = np.array([[1+self.lam*self.theta_st*self.alpha, 0], [self.lam * (self.theta_st**2) / 2, 1]])

            elif self.env.params_dict["Flux"] == "LF":
                pass


        if self.env.params_dict["Theta_choice_method"] == "Smoothing":
            x = self.tanh_factor
            J = np.array([[1 - 25 * self.v[i] * self.lam * self.sech(100*(-.5+ self.v[i]/self.w[i]))**2 / self.w[i] + self.lam * (.75 + .25*np.tanh(x*(-.5 + self.v[i]/self.w[i]))) ,
                           25 * self.lam * self.sech(100*(-.5 + self.v[i]/self.w[i]))**2],
                          [-(25 * self.v[i] * self.lam * self.sech(100*(-.5 + self.v[i]/self.w[i]))**2 * (.75 + .25*np.tanh(x*(-.5 + self.v[i]/self.w[i])))) / self.w[i] + .5 * self.lam * (.75 + .25*np.tanh(x * (-.5 + self.v[i]/self.w[i])))**2 ,
                           1 + 25 * self.lam * self.sech(100 * (-.5 + self.v[i]/self.w[i]))**2 * (3/4 + (1/4)*np.tanh(x * (-.5 + self.v[i]/self.w[i])))]])
            #print(np.linalg.norm(J))
        return J


    def Newton(self, epsilon=1e-6, maxiter=10):
        w_ = np.copy(self.u_down)
        w_[0] = 0
        #v = np.empty_like(self.u_down)
        self.v = self.u_til - self.u_down
        self.v[0] = 0 #
        self.w = np.zeros_like(w_)
        #(Neumann at the left)
        iter=0

        while np.linalg.norm(self.w - w_) > epsilon and maxiter>iter:
            for i in range(1,self.env.mesh.Nx +1):
                w_[i] = self.w[i]

                near0_eps = 1e-12
                #if w[i] != 0:
                if np.abs(self.w[i]) > near0_eps:
                    #update thetas 
                    self.theta_choice(i, epsilon=self.env.params_dict["Theta_choice_epsilon"])
                    #compute step (linear system resolution)
                    s = np.linalg.solve(self.compute_J(i, epsilon=self.env.params_dict["Theta_choice_epsilon"]), -self.F(w,v,i))
                else:
                    self.thetas[i] = 1.
                    s = np.linalg.solve(np.array([[1+self.lam,0],[self.lam/2,1]]), -self.F(i))

                #iterate Newton problem
                self.w[i] += s[0]
                self.v[i] += s[1]
            #print(np.mean(w))
            iter += 1
        
        if self.env.params_dict["print_Newton_iter"]==True:
            print(iter, end=" ")

        for i in range(self.w.size):
            self.theta_choice(i, epsilon=self.env.params_dict["Theta_choice_epsilon"])


    def update_thetas(self):

        if self.theta_computation == "Newton":
            self.Newton(epsilon=self.env.params_dict["Method_convergence_epsilon"])

        else:
            print("Wrong theta choice method type")


    def SATh_Scheme(self):

        t = 0
        self.thetas = np.ones(self.env.mesh.Nx+1) * self.theta_st
        self.u_down = self.env.funcs.init_sol.copy()
        self.u_up = np.empty_like(self.u_down)
        self.u_up[0] = self.u_down[0]

        if self.env.params_dict["Timer"]==True:
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
            self.u_til = self.v + self.u_up   #update u_til

            t += self.env.mesh.dt
            if self.env.params_dict["Timer"]==True:
                self.timer()

        return self.u_up

