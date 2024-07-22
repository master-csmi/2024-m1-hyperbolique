import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
from scipy.optimize import newton_krylov, root
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
        "cfl": 5,  #:0 for a mesh built with Nt
        "dt": None, 
        "mesh_type": "offset_constantDx",

        #Type of problem and parameters of the associated functions
        "Problem": "Linear_advection", #"Burgers"
        "coefficients": [1.], #coefficients of the equations.
                              # -> Linear advection: [a (->speed => alpha)]
                              # -> RIPA: ...

        "Boundary": "dirichlet", #/"periodic"
        "init_func": "jump1", #/ "jump2", "bell", "sine_shock" ; for RIPA: "smooth", "flat", "nonflat" (see 5.1,5.2,5.3 of Desveaux et al. paper)
        "init_function_params": [0.1, 0., 1.], # [d_0, ... d_M, u_0, ... u_N]
                                       # d_i : Location(s) of the singularities/bell (left to right),
                                       # u_j : values of the constant segments (left to right)

        #Type of Scheme
        "Scheme": "SATh", #/"Theta"
        "Flux": "LF", #/"Upwind"
        "Scheme_tol": 6e-6,
        "Scheme_maxiter": 30,

            #Theta-parameters
            "Theta_st": 0.5,
            "Theta_min": 0.5,
            "Theta_max": 1,   #the maximum value we want theta to be able to take (when using 'MinMax')
            
            #Methods for Thetas computation and related parameters
            "Newton_convergence_epsilon": 6e-6,
            "Newton_maxiter":100,
            "Theta_choice_method": "MinMax", #/"Smoothing" /->"MinMax" is the discontinuous function
                #Parameter to control the Smoothing method Function:
                "kappa": 10,
                #Parameter for MinMax method threshold:
                "Theta_choice_epsilon": 1e-100,#1e-6,

        #Others
        "Exact": True, #True if you want to compute the exact solution. Switch to False when trying to launch a computation with special initial function or parameters
        "Auto_launch_computation": True,  #Switch to False if you just want to access to this Library's methods without launching a computation.
        "Loading_bar":False,
        "print_Newton_iter":False,
        "Path": "../pictures",
        "Animation": False,
        }

        #replacing by input values in the dict:
        for key in kwargs.keys():
            self.params_dict[key] = kwargs[key]

        if self.params_dict["Problem"]=="RIPA":
            self.dim = 3
        else:
            self.dim = 1

        
        self.cfl = self.params_dict["cfl"]
        t = self.cfl
        t_type="cfl"

        self.tf = self.params_dict["tf"]
        self.coefs = self.params_dict["coefficients"]

        self.mesh = Mesh(self.params_dict["a"], self.params_dict["b"],
                         self.params_dict["Nx"], self.tf, t, t_type, 
                         self.params_dict["Boundary"], self.params_dict["mesh_type"])
        self.funcs = Functions(self.mesh, self.params_dict["Problem"],
                               self.params_dict["init_function_params"],
                               self.tf, self.params_dict["init_func"],
                               self.params_dict["Exact"])
        
        if self.params_dict["Problem"]=="Linear_advection":
            self.alpha = np.abs(self.coefs[0])
        else:
            self.alpha = self.compute_alpha(self.df(self.funcs.init_sol))
        self.mats = Matrices(self.mesh, self.params_dict["Boundary"], self.params_dict["coefficients"][0])

        if self.params_dict["Problem"] != "Linear_advection" and self.params_dict["Flux"] == "Upwind":
            raise ValueError("The Backwards Euler / Upwind scheme is not fit to give a good numerical solution to a nonlinear problem")

        if self.params_dict["Scheme"] == "Theta" and self.params_dict["Flux"]=="Upwind":  #Simple/Standard Theta Scheme
            self.theta = self.params_dict["Theta_st"]
            scheme = Theta_Scheme(self)
            if self.params_dict["Auto_launch_computation"]==True:
                self.sol_num = scheme.solver()
        
        else:
            self.solver = SATh_Solver(self)
            if self.params_dict["Auto_launch_computation"]==True:
                self.sol_num = self.solver.SATh_Scheme(tol=self.params_dict["Scheme_tol"],
                                                       #maxiter=self.params_dict["Scheme_maxiter"])
                                                        )

    def print_params(self):
        return self.params_dict

    def sol_ex(self):
        return self.funcs.exact(self.mesh.nodes, 
                                self.params_dict["init_function_params"], 
                                self.tf)

    def compute_alpha(self, df_u):  #compute the max propagation speed of the problem, at one time step
        if self.params_dict["Problem"] == "Burgers":
            return max(np.abs(df_u))
        elif self.params_dict["Problem"] == "RIPA":
            [h,u,T] = self.funcs.init_sol
            g = 1
            return max(max(np.abs(np.sqrt(g*T*h)-u)), max(np.abs(np.sqrt(g*T*h)+u)))

    def f(self, u):
        if self.params_dict["Problem"] == "Linear_advection":
            return self.alpha * u
        elif self.params_dict["Problem"] == "Burgers":
            return u**2 / 2
        elif self.params_dict["Problem"] == "RIPA":
            [h,u,T] = self.funcs.init_sol
            g = 1
            return np.array([h*u, h*u**2 + g*T*h**2/2, h*T*u])

    def df(self, u):
        if self.params_dict["Problem"] == "Linear_advection":
            return self.alpha
        elif self.params_dict["Problem"] == "Burgers":
            #self.alpha=... -> updated at each step
            return u
        elif self.params_dict["Problem"] == "RIPA":
            #self.alpha=... -> updated at each step
            return np.empty_like(u)


class SATh_Solver:
    def __init__(self, env):
        self.env = env
        self.dim = env.dim

        self.theta_st = env.params_dict["Theta_st"]
        self.theta_min = env.params_dict["Theta_min"]
        self.thetas = np.ones(shape=(self.dim,env.mesh.Nx+1)) * self.theta_st
        self.dthetas = np.zeros_like(self.thetas)

        self.u_down = self.env.funcs.init_sol.copy()
        self.u_up, self.u_til = np.empty_like(self.u_down), np.empty_like(self.u_down)
        self.bound_vals = [self.env.funcs.init_sol[0],self.env.funcs.init_sol[-1]]

        self.w, self.v = np.empty_like(self.u_down), np.empty_like(self.u_down)
        self.f, self.df = self.env.f, self.env.df

        self.kappa = self.env.params_dict["kappa"]
        self.near0_eps = 1e-14

        self.alpha = env.alpha

        if env.params_dict["Flux"]=="Upwind":
            self.lam = env.mesh.dt/env.mesh.dx 
        elif env.params_dict["Flux"]=="LF":
            self.lam = .5 * env.mesh.dt/env.mesh.dx
        else: print("Unknown Flux type")

        self.Th = Theta_Managing(self)  #The functions associated to the choices of thetas are stored away


    def timer(self, action=None):
        if action == "set":
            print(int(self.env.tf//self.env.mesh.dt)*int(self.alpha) * "_")
        else:
            print("-",end="")


    def doubletab(self, xs):
        if self.dim == 1:
            [x1,x2]=xs
            if x1.shape != x2.shape:
                raise TabError("x1 & x2 must have the same shape")
            new = np.empty(shape=(x1.shape[0]*2))
            new[:-1:2], new[1::2] = x1, x2

        else:
            if not isinstance(xs, np.ndarray):
                xs = np.array(xs)
            d = self.dim
            if 2*d!=xs.shape[0]:
                raise TabError("Wrong number of arrays")
            for i in range(xs.shape[0]-1):
                if xs[i].shape != xs[i+1].shape:
                    raise TabError("x1 & x2 must have the same shape")
            new = np.empty(shape=(xs[0].shape[0]*2*d))

            j = 0
            for i in range(xs[0].shape[0]):
                for k in range(self.dim*2):
                    new[j+k] = xs[k][i]
                j += self.dim*2

        return new

    

    def F(self, X):
        f = self.f
        d = self.dim

        w_x = X[:-1:2]
        w = self.doubletab([w_x]*2)
        _w = self.doubletab([np.roll(w_x,1)]*2)
        w_ = self.doubletab([np.roll(w_x,-1)]*2)


        Thetas = self.doubletab([self.thetas, self.thetas**2]*d)
        _Thetas = self.doubletab([np.roll(self.thetas,1),np.roll(self.thetas**2,1)]*d)
        Thetas_ = self.doubletab([np.roll(self.thetas,-1),np.roll(self.thetas**2,-1)]*d)

        if d==1:
            u = self.doubletab([self.u_down,self.u_down])
            _u = self.doubletab([np.roll(self.u_down,1),np.roll(self.u_down,1)])
            u_ = self.doubletab([np.roll(self.u_down,-1),np.roll(self.u_down,-1)])
        else:
            u = np.empty(shape=(2 * d * (self.env.mesh.Nx+1)))
            _u = np.empty_like(u)
            u_ = np.empty_like(u)
            j=0
            for i in range(d):
                u[j:-(d-j):d], u[j:-(d-j):d] = self.u_down[i], self.u_down[i]
                u[j+1:-(d-j+1):d], u[j+1:-(d-j+1):d] = self.u_down[i], self.u_down[i]
                j+=2

        lams = np.ones(shape=self.thetas.shape) * self.lam
        lam_2 = self.doubletab([lams,lams/2]*self.dim)


        if self.env.params_dict["Flux"] == "LF":
            if self.env.params_dict["Boundary"]=="periodic":
                ret = X + lam_2 * (Thetas_ * (f(w_+u_) - f(u_) - self.alpha*w_) + 2*self.alpha*Thetas * w - _Thetas * (
                    f(_w+_u) - f(_u) + self.alpha*_w)) - (
                    -lam_2 * (f(u_)-f(_u)) + lam_2*self.alpha*(u_-2*u+_u))
            elif self.env.params_dict["Boundary"]=="dirichlet":
                ret = np.zeros_like(u)
                ret[2*d:-2*d] = X[2*d:-2*d] + lam_2[2*d:-2*d] * (Thetas_[2*d:-2*d] * (f(w_[2*d:-2*d]+u_[2*d:-2*d]) - f(u_[2*d:-2*d]) - self.alpha*w_[2*d:-2*d]) 
                                                     + 2*self.alpha*Thetas[2*d:-2*d] * w[2*d:-2*d] - _Thetas[2*d:-2*d] * (
                    f(_w[2*d:-2*d]+_u[2*d:-2*d]) - f(_u[2*d:-2*d]) + self.alpha*_w[2*d:-2*d])) - (
                    -lam_2[2*d:-2*d] * (f(u_[2*d:-2*d])-f(_u[2*d:-2*d])) + lam_2[2*d:-2*d]*self.alpha*(u_[2*d:-2*d]-2*u[2*d:-2*d]+_u[2*d:-2*d]))

        elif self.env.params_dict["Flux"] == "Upwind":
            ret = np.zeros_like(u)
            ret[2*d:] = X[2*d:] + lam_2[2*d:] * Thetas[2*d:] * (f(w[2*d:] + u[2*d:]) - f(u[2*d:])) - (
                        -lam_2[2*d:] * (f(u[2*d:]) - _Thetas[2*d:] * (f(_w[2*d:] + _u[2*d:])-f(_u[2*d:])) - f(_u[2*d:])))


        if self.env.params_dict["Scheme"]=="SATh":  #the update must be included in the function
            for i in range(self.thetas.shape[0]):
                self.Th.theta_choice(self.thetas, i,
                                    epsilon=self.env.params_dict["Theta_choice_epsilon"])

        return ret


    def Newton(self, epsilon=1e-6, maxiter=10):
        w_ = np.copy(self.u_down)
        w_[0] = 0
        self.v = self.u_til - self.u_down
        self.v[0] = 0
        self.w = np.zeros_like(w_)
        iter=0

        self.storew = self.w.copy()
        self.storev = self.v.copy()
        #X = self.doubletab(self.w,self.v)
        O = np.zeros_like(self.v)
        X = self.doubletab([O,self.v]*self.dim)
        self.thetas = np.ones_like(self.thetas)*self.theta_st#
        X = newton_krylov(self.F, X,
                            iter=maxiter,
                            verbose=False,
                            f_tol=epsilon)
        """r = root(self.F, X, tol=epsilon)
        X = r.x"""
        #last update of w and v
        self.w = X[:-1:2]
        self.v = X[1::2]

        self.F_ = self.F(X)

        for i in range(self.w.size):
            self.Th.theta_choice(self.thetas, i,
                                epsilon=self.env.params_dict["Theta_choice_epsilon"])
        self.Th.dthetas_update(self.dthetas)
        
        if self.env.params_dict["print_Newton_iter"]==True:
            print(iter, end=" ")
    

    def ret_F(self):
        return self.F_
    

    def SATh_Scheme(self, tol=6e-6):

        t = 0
        self.thetas = np.ones(self.env.mesh.Nx+1) * self.theta_st
        self.u_down = self.env.funcs.init_sol.copy()
        self.u_up = np.empty_like(self.u_down)
        #self.u_up[0] = self.u_down[0]
        dt = self.env.mesh.dt

        if self.env.params_dict["Loading_bar"]==True:
            self.timer("set")
        
        if self.env.params_dict["Animation"]==True:
                self.thetas_save = []
                self.numsol_save = []

        while (t<self.env.tf):

            if self.env.params_dict["Problem"] != "Linear_advection":
                    self.alpha = self.env.compute_alpha(self.df(self.u_up))
                    dt = self.env.mesh.dt / self.alpha
                    if self.env.params_dict["Flux"]=="Upwind":
                        self.lam = dt/self.env.mesh.dx
                    elif self.env.params_dict["Flux"]=="LF":
                        self.lam = .5 * dt/self.env.mesh.dx
                

            if self.env.params_dict["Scheme"]=="Theta":

                if self.env.params_dict["Flux"]=="Upwind":
                    A = self.env.mats.Iter_Mat(self.env.mesh,
                                        self.thetas,
                                        self.alpha,
                                        adaptive=True,
                                        flux = self.env.params_dict["Flux"],
                                        boundary=self.env.params_dict["Boundary"])
                    coefs = self.env.mesh.dt*(np.eye(self.env.mesh.Nx+1)-np.diag(self.thetas))
                    b = self.u_down - self.env.alpha * coefs@(self.env.mats.Dx @ self.u_down)
                    b[0] = self.u_down[0]
                    self.u_up, _ = gmres(A, b)
                    self.w = self.u_up - self.u_down

                elif self.env.params_dict["Flux"]=="LF":
                    
                    #X = self.doubletab(self.w,self.v)
                    O = np.zeros_like(self.v)
                    X = self.doubletab([O,self.v]*self.dim)
                    self.w = newton_krylov(self.F, X,
                                            #iter=self.env.params_dict["Scheme_maxiter"],
                                            f_tol=tol,
                                            verbose=False)[:-1:2]
                    self.u_up = self.u_down + self.w


            else:
                self.Newton(epsilon=self.env.params_dict["Newton_convergence_epsilon"],
                            maxiter=self.env.params_dict["Newton_maxiter"])
                self.u_up = self.u_down + self.w
                

            self.u_down = self.u_up.copy()
            #print(self.env.params_dict["Scheme"], np.linalg.norm(self.u_down))
            self.u_til = self.v + self.u_up   #update u_til

            t += dt
            if self.env.params_dict["Loading_bar"]==True:
                self.timer()

            if self.env.params_dict["Animation"]==True:
                self.thetas_save.append(self.thetas.copy())
                self.numsol_save.append(self.u_up.copy())

        if self.env.params_dict["Animation"]==True:
            self.thetas_save = np.array(self.thetas_save)
            self.numsol_save = np.array(self.numsol_save)

        return self.u_up

