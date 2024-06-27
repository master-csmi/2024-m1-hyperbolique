import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov
import matplotlib.pyplot as plt

from Library.utilities import *


class Problem():
    def __init__(self, **kwargs):

        self.params_dict = {
        #Numerical domain
        "a": 0,
        "b": 1,
        "Nx": 500,
        "tf": 0.1,
        "cfl": 5,
        "mesh_type": "offset_constantDx",

        #Type of problem and parameters of the associated functions
        "f": "linear_advection",
        "alpha": 1,
        "Boundary": "constant", #"constant"=Dirichlet
        "init_func": "jump1",
        "init_function_params": [0.2], #Location of the singularity/bell

        #Type of Scheme
        "Scheme": "SATh", #/"Theta"
        "Flux": "UP", #/"LF"

            #Theta-parameters
            "Theta_st": 0.5,
            "Theta_min": 0.5,
            "Theta_max": 1,   #This is only useful for plots with different theta values, with the simple theta scheme
            
            #Methods for Thetas computation and related parameters
            "Theta_solving_method": "Newton", #"Fixed-Point" has been abandoned
            "Method_convergence_epsilon": 1e-6,
            "Theta_choice_method": "MinMax", #/"Smoothing" /->"MinMax" is the discontinuous function
                #Parameter to control the Smoothing method Function:
                "kappa": 10,
                #Parameter for MinMax method threshold:
                "Theta_choice_epsilon": 1e-6,
            "Newton_solver": "LO_NewtonKrylov", #"Jacobian", #/"LO_NewtonKrylov"-> for Matrix-free Newton Krylov Method (using linear operators)
                #In the case of SATh-UP (with "Jacobian" solving method):
                "Jacobian_method": "Classic", #/"Smoothing"

        #Others
        "Timer":False,
        "print_Newton_iter":False,
        "Path": "../pictures",
        }

        #replacing by input values in the dict:
        for key in kwargs.keys():
            self.params_dict[key] = kwargs[key]

        self.tf = self.params_dict["tf"]
        self.alpha = self.params_dict["alpha"]
        self.mesh = Mesh(self.params_dict["a"], self.params_dict["b"], 
                         self.params_dict["Nx"], self.params_dict["cfl"], 
                         self.params_dict["Boundary"], self.params_dict["mesh_type"])
        self.mats = Matrices(self.mesh, self.params_dict["Boundary"], self.alpha)
        self.funcs = Functions(self.mesh, self.params_dict["init_function_params"],
                               self.tf, self.params_dict["init_func"])

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
        return self.funcs.exact(self.mesh.nodes, 
                                self.params_dict["init_function_params"], 
                                self.tf)
    
    def f(self, u):
        if self.params_dict["f"] == "linear_advection":
            return self.alpha * u
        
    def df(self, u):
        if self.params_dict["f"] == "linear_advection":
            return self.alpha



class SATh_Solver:
    def __init__(self, env):
        self.env = env
        self.theta_st = env.params_dict["Theta_st"]
        self.theta_min = env.params_dict["Theta_min"]
        self.thetas = np.ones(env.mesh.Nx+1) * self.theta_st
        self.dthetas = np.zeros_like(self.thetas)
        self.theta_computation = env.params_dict["Theta_solving_method"]#

        self.u_down = self.env.funcs.init_sol.copy()
        self.u_up, self.u_til = np.empty_like(self.u_down), np.empty_like(self.u_down)
        self.bound_vals = [self.env.funcs.init_sol[0],self.env.funcs.init_sol[-1]]

        self.w, self.v = np.empty_like(self.u_down), np.empty_like(self.u_down)
        self.f, self.df = self.env.f, self.env.df

        self.kappa = self.env.params_dict["kappa"]
        self.near0_eps = 1e-14

        if env.params_dict["Flux"]=="UP":
            self.alpha = self.env.alpha
            self.lam = env.mesh.dt/env.mesh.dx 
        elif env.params_dict["Flux"]=="LF":
            #self.alpha = self.alpha_LF()
            self.alpha = self.env.alpha
            self.lam = .5 * env.mesh.dt/env.mesh.dx
        else: print("Unknown Flux type")

        self.Th = Theta_Managing(self)  #The functions associated to the choices of thetas are stored away

        #
        if self.env.params_dict["Flux"]=="LF" and self.env.params_dict["Jacobian_method"]=="Smoothing":
            raise ValueError("Jacobian method 'Smoothing' is not compatible with Lax-Friedrichs")
        #


    def timer(self, action=None):
        if action == "set":
            print(int(self.env.tf//self.env.mesh.dt) * "_")
        else:
            print("-",end="")

    
    def LF_Newton_Matrices(self, block, i):
        #This function is used to build the Jacobian matrix for the "Jacobian" Newton method
        mat = np.empty(shape=(2,2))

        if block=="A":
            if np.abs(self.w[i-1])<self.near0_eps:
                mat[0,0] = -self.lam * (self.thetas[i-1] * (self.df(self.w[i-1] + self.u_up[i-1])+self.alpha))
                mat[0,1] = 0
                mat[1,0] = self.lam * (-.5*self.thetas[i-1]**2 * (self.df(self.w[i-1] + self.u_up[i-1]) + self.alpha))
                mat[1,1] = 0
            else:
                mat[0,0] = -self.lam * (self.thetas[i-1] * (self.df(self.w[i-1] + self.u_up[i-1])+self.alpha)
                                + (self.v[i-1]/(self.w[i-1]**2)) * self.dthetas[i-1] * (self.alpha * self.w[i-1] + self.f(self.w[i-1] + self.u_up[i-1]) - self.f(self.u_up[i-1])))
                mat[0,1] = -(self.lam / self.w[i-1]) * (-self.f(self.u_up[i-1]) + self.alpha*self.w[i-1] + self.f(self.w[i-1] + self.u_up[i-1])) * self.dthetas[i-1]
                mat[1,0] = self.lam * (-.5*self.thetas[i-1]**2 * (self.df(self.w[i-1] + self.u_up[i-1]) + self.alpha)
                                + (self.v[i-1]/(self.w[i-1]**2)) * self.thetas[i-1] * self.dthetas[i-1] * (self.alpha*self.w[i-1] + self.f(self.w[i-1]+self.u_up[i-1]) - self.f(self.u_up[i-1])))
                mat[1,1] = -self.lam * (1/self.w[i-1]) * (-self.f(self.u_up[i-1]) + self.alpha*self.w[i-1] + self.f(self.w[i-1]+self.u_up[i-1])) * self.thetas[i-1] * self.dthetas[i-1]

        elif block=="B":
            if np.abs(self.w[i])<self.near0_eps:
                mat[0,0] = 1 + 2*self.alpha*self.lam*(self.thetas[i])
                mat[0,1] = 0
                mat[1,0] = 2*self.alpha*self.lam*(.5*self.thetas[i]**2)
                mat[1,1] = 1
            else:
                mat[0,0] = 1 + 2*self.alpha*self.lam*(self.thetas[i] - (self.v[i]/self.w[i]) * self.dthetas[i])
                mat[0,1] = 2*self.alpha*self.lam*self.dthetas[i]
                mat[1,0] = 2*self.alpha*self.lam*(.5*self.thetas[i]**2 - (self.v[i]/self.w[i]) * self.thetas[i] * self.dthetas[i])
                mat[1,1] = 1 + 2*self.alpha*self.lam*self.thetas[i]*self.dthetas[i]


        elif block=="C":
            if np.abs(self.w[i+1]<self.near0_eps):
                mat[0,0] = self.lam * (self.thetas[i+1] * (self.df(self.w[i+1] + self.u_up[i+1]) - self.alpha))
                mat[0,1] = 0
                mat[1,0] = self.lam * (.5*self.thetas[i+1]**2 * (self.df(self.w[i+1] + self.u_up[i+1]) - self.alpha))
                mat[1,1] = 0
            else:
                mat[0,0] = self.lam * (self.thetas[i+1] * (self.df(self.w[i+1] + self.u_up[i+1]) - self.alpha)
                                + (self.v[i+1]/(self.w[i+1]**2)) * self.dthetas[i+1] * (self.alpha * self.w[i+1] - self.f(self.w[i+1] + self.u_up[i+1]) + self.f(self.u_up[i+1])))
                mat[0,1] = -(self.lam / self.w[i+1]) * (self.f(self.u_up[i+1]) + self.alpha*self.w[i+1] - self.f(self.w[i+1] + self.u_up[i+1])) * self.dthetas[i+1]
                mat[1,0] = self.lam * (.5*self.thetas[i+1]**2 * (self.df(self.w[i+1] + self.u_up[i+1]) - self.alpha)
                                + (self.v[i+1]/(self.w[i+1]**2)) * self.thetas[i+1] * self.dthetas[i+1] * (self.alpha*self.w[i+1] - self.f(self.w[i+1]+self.u_up[i+1]) + self.f(self.u_up[i+1])))
                mat[1,1] = -self.lam * (1/self.w[i+1]) * (self.f(self.u_up[i+1]) + self.alpha*self.w[i+1] - self.f(self.w[i+1]+self.u_up[i+1])) * self.thetas[i+1] * self.dthetas[i+1]

        return mat


    def F_mat(self, i):
        #This function is used to build the matrix corresponding to the function F for the "Jacobian" Newton method
        if self.env.params_dict["Flux"]=="UP":
            return np.array([
                self.w[i] + self.lam * self.thetas[i] * self.w[i] + self.lam * (self.f(self.u_up[i]) - self.thetas[i-1]*self.w[i-1] - self.f(self.u_up[i-1])),
                self.v[i] + self.lam * .5 * self.thetas[i]**2 * self.w[i] + self.lam * .5 * (self.f(self.u_up[i]) - self.thetas[i-1]**2 * self.w[i-1] - self.f(self.u_up[i-1]))
            ])

        elif self.env.params_dict["Flux"]=="LF":
            F = np.empty(shape=(2*(self.env.mesh.Nx+1)))
            i=0
            while i<2*(self.env.mesh.Nx):
                j = i//2
                F[i] = self.w[j] + self.lam * (self.thetas[j+1] * (self.f(self.w[j+1] + self.u_up[j+1]) - self.f(self.u_up[j+1]) - self.alpha*self.w[j+1])
                                        + 2*self.alpha*self.thetas[j]*self.w[j] - self.thetas[j-1] * (
                                            self.f(self.w[j-1] + self.u_up[j-1]) - self.f(self.u_up[j-1]) + self.alpha*self.w[j-1])) 
                + self.lam * ((self.f(self.u_up[j+1]) - self.f(self.u_up[j-1])) - self.alpha*(self.u_up[j+1] - 2*self.u_up[j] + self.u_up[j-1]))

                F[i+1] = self.v[j] + self.lam * ((.5*self.thetas[j+1]**2) * (self.f(self.w[j+1] + self.u_up[j+1]) - self.f(self.u_up[j+1]) - self.alpha*self.w[j+1])
                                        + 2*self.alpha*(.5*self.thetas[j]**2)*self.w[j] - (.5*self.thetas[j-1]**2) * (
                                            self.f(self.w[j-1] + self.u_up[j-1]) - self.f(self.u_up[j-1]) + self.alpha*self.w[j-1]))
                + .5*self.lam * ((self.f(self.u_up[j+1]) - self.f(self.u_up[j-1])) - self.alpha*(self.u_up[j+1] - 2*self.u_up[j] + self.u_up[j-1]))

                i += 2

            return F

    def doubletab(self, x1, x2):
            if x1.shape != x2.shape:
                raise TabError("x1 & x2 must have the same shape")
            new = np.empty(shape=(x1.shape[0]*2))
            new[:-1:2], new[1::2] = x1, x2
            return new

    def F(self, X):   #For LF using "LO_NewtonKrylov" method
        f = self.f

        w_x = X[:-1:2]
        w = self.doubletab(w_x, w_x)
        _w = self.doubletab(np.roll(w_x,1),np.roll(w_x,1))
        w_ = self.doubletab(np.roll(w_x,-1),np.roll(w_x,-1))
        u = self.doubletab(self.u_up,self.u_up)
        _u = self.doubletab(np.roll(self.u_up,1),np.roll(self.u_up,1))
        u_ = self.doubletab(np.roll(self.u_up,-1),np.roll(self.u_up,-1))
        Thetas = self.doubletab(self.thetas, .5*self.thetas**2)
        _Thetas = self.doubletab(np.roll(self.thetas,1),np.roll(.5*self.thetas**2,1))
        Thetas_ = self.doubletab(np.roll(self.thetas,-1),np.roll(.5*self.thetas**2,-1))
        lams = np.ones(shape=self.thetas.shape) * self.lam
        lam_2 = self.doubletab(lams,lams/2)

        ret = u + self.lam * (Thetas_ * (f(w_+u_) - f(u_) - self.alpha*w_) + 2*self.alpha*Thetas * w - _Thetas * (
              f(_w+_u) - f(_u) + self.alpha*_w)) - (
              -lam_2 * (f(u_)-f(_u)) + lam_2*self.alpha*(u_-2*u+_u))
        #Boundary conditions:
        ret[0] = self.storew[0]
        ret[1] = self.storev[0]
        ret[-2] = self.storew[-1]
        ret[-1] = self.storev[-1]
        """ret[0] = 0
        ret[1] = 0
        ret[-2] = 0
        ret[-1] = 0"""

        #updating for the next iteration of newton_krylov
        self.w = ret[:-1:2]
        self.v = ret[1::2]
        for i in range(self.thetas.shape[0]):
            self.Th.theta_choice(self.thetas, i,
                                 epsilon=self.env.params_dict["Theta_choice_epsilon"])

        return ret


    def compute_J(self, i, epsilon=1e-6):    #Compute the Jacobian
        if self.env.params_dict["Jacobian_method"] == "Classic":
            if self.env.params_dict["Flux"]=="UP":
                if self.env.params_dict["f"] == "linear_advection":
                    if np.abs(self.w[i])>=epsilon :
                        if self.v[i]/self.w[i] > self.theta_min:
                            J = np.array([[1, self.alpha * self.lam],
                                          [-(self.v[i]**2)*self.alpha*self.lam / (2*(self.w[i]**2)),
                                           1 + self.alpha*self.lam/self.w[i]]])
                        else :
                            J = np.array([[1+self.lam*self.theta_min*self.alpha, 0],
                                          [self.lam * self.alpha*(self.theta_min**2) / 2, 1]])
                    else :
                        J = np.array([[1+self.lam*self.theta_st*self.alpha, 0],
                                      [self.lam * (self.theta_st**2) / 2, 1]])


            elif self.env.params_dict["Flux"] == "LF":
                J = np.zeros(shape=(2*(self.env.mesh.Nx+1),2*(self.env.mesh.Nx+1)))
                i=0
                while i < 2*(self.env.mesh.Nx+1):
                    j = i//2
                    if i!=0:
                        J[i:i+2,i-2:i] = self.LF_Newton_Matrices(block="A",i=j)
                    J[i:i+2,i:i+2] = self.LF_Newton_Matrices(block="B",i=j)
                    if i!=2*self.env.mesh.Nx:
                        J[i:i+2,i+2:i+4] = self.LF_Newton_Matrices(block="C",i=j)
                    i += 2

        elif self.env.params_dict["Jacobian_method"] == "Smoothing":
            k = self.kappa
            if np.abs(self.w[i])>=self.near0_eps:
                J = np.array([[1 - (1/4)* k * self.v[i] * self.lam * self.Th.sech(k*(-.5+ self.v[i]/self.w[i]))**2 / self.w[i] + (
                                self.lam * (.75 + .25*np.tanh(k*(-.5 + self.v[i]/self.w[i])))) ,

                            (1/4) * k * self.lam * self.Th.sech(k*(-.5 + self.v[i]/self.w[i]))**2],

                            [-((1/4)*k * self.v[i] * self.lam * self.Th.sech(k*(-.5 + self.v[i]/self.w[i]))**2 * (
                                .75 + .25*np.tanh(k*(-.5 + self.v[i]/self.w[i])))) / self.w[i] + (
                                .5 * self.lam * (.75 + .25*np.tanh(k * (-.5 + self.v[i]/self.w[i])))**2) ,

                            1 + (1/4)*k * self.lam * self.Th.sech(k * (-.5 + self.v[i]/self.w[i]))**2 * (3/4 + (1/4)*np.tanh(k * (-.5 + self.v[i]/self.w[i])))]])
                
                #print(np.linalg.norm(J))
            else:
                self.thetas[i] = 1.
                J = np.array([[1+self.lam,0],[self.lam/2,1]])

        return J


    def Newton(self, epsilon=1e-6, maxiter=10):
        w_ = np.copy(self.u_down)
        w_[0] = 0
        self.v = self.u_til - self.u_down
        self.v[0] = 0
        self.w = np.zeros_like(w_)
        iter=0

        if self.env.params_dict["Newton_solver"] == "LO_NewtonKrylov" and self.env.params_dict["Flux"]!="UP":
            self.storew = self.w.copy()
            self.storev = self.v.copy()
            X = self.doubletab(self.w,self.v)
            #X = np.ones(shape=(self.w.shape[0]*2))
            X = newton_krylov(self.F, X,
                              iter=maxiter,
                              verbose=False,
                              f_tol=epsilon)
            #update the thetas:
            for i in range(self.w.size):
                self.Th.theta_choice(self.thetas, i,
                                     epsilon=self.env.params_dict["Theta_choice_epsilon"])
            self.Th.dthetas_update(self.dthetas)

        """#Does not currently work
        elif self.env.params_dict["Newton_solver"] == "Jacobian" or self.env.params_dict["Flux"]=="UP":

            while np.linalg.norm(self.w - w_) >= epsilon and iter<maxiter:

                if self.env.params_dict["Flux"]=="UP":
                    for i in range(1,self.env.mesh.Nx +1):
                        w_[i] = self.w[i]

                        self.Th.theta_choice(self.thetas, i,
                                             epsilon=self.env.params_dict["Theta_choice_epsilon"])
                        s = np.linalg.solve(self.compute_J(i, 
                                                           epsilon=self.env.params_dict["Theta_choice_epsilon"]),
                                                           -self.F_mat(i))

                        self.w[i] += s[0]
                        self.v[i] += s[1]
            
                elif self.env.params_dict["Flux"]=="LF":
                    w_ = self.w

                    #if np.mean(self.thetas)!=1.:
                    #    print("thetas not 1, iter:", iter, "value mean thetas:", np.mean(self.thetas))
                    self.Th.dthetas_update(self.dthetas)
                    #boundary conditions:
                    J = self.compute_J(0, epsilon=self.env.params_dict["Theta_choice_epsilon"])
                    J[0], J[1], J[-2], J[-1] = 0,0,0,0
                    #compute step (linear system resolution)
                    s, _ = gmres(J, -self.F_mat(0))
                    

                    #print("max s",np.max(s), "argmax s",np.argmax(s), "s mean",np.mean(s))
                    self.w += s[::2]
                    self.v += s[1::2]
                    
                    for i in range(self.thetas.shape[0]):
                        self.Th.theta_choice(self.thetas, i,
                                             epsilon=self.env.params_dict["Theta_choice_epsilon"])
                    
                #update the thetas:
                for i in range(self.w.size):
                    self.Th.theta_choice(self.thetas, i,
                                         epsilon=self.env.params_dict["Theta_choice_epsilon"])
                self.Th.dthetas_update(self.dthetas)

                print(iter)
                iter += 1
                if iter == maxiter:
                    print("Newton stopped at maxiter")
        
            
            if self.env.params_dict["print_Newton_iter"]==True:
                print(iter, end=" ")

        else:
            raise ValueError("Error in the values of 'Flux' and/or 'Newton_solver'.")
        """

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
                self.Newton(epsilon=self.env.params_dict["Method_convergence_epsilon"])
            #print(min(self.thetas), max(self.thetas))
            
            coefs = self.env.mesh.dt*(np.eye(self.env.mesh.Nx+1)-np.diag(self.thetas))
            A = self.env.mats.Iter_Mat(self.env.mesh,
                                       self.thetas,
                                       self.env.alpha,
                                       adaptive=True)
            #A = sparse.linalg.LinearOperator((self.env.mesh.Nx+1,self.env.mesh.Nx+1), matvec=self.env.funcs.Iter_Func)
            b = self.u_down - self.env.alpha * coefs@(self.env.mats.Dx @ self.u_down)
            b[0] = self.u_down[0]

            self.u_up, _ = gmres(A, b)
            self.u_down = self.u_up.copy()
            self.u_til = self.v + self.u_up   #update u_til

            t += self.env.mesh.dt
            if self.env.params_dict["Timer"]==True:
                self.timer()

        return self.u_up

