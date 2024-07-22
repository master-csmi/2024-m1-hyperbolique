import numpy as np
import scipy.sparse as sparse


class Theta_Managing:
    def __init__(self, solver):
        self.solver = solver
        self.kappa = self.solver.kappa
        self.method = self.solver.env.params_dict["Theta_choice_method"]
        self.near0_eps = self.solver.near0_eps

    def sech(self, x):
        if np.abs(x) > 20:  #in order to avoid overflow warnings
            return 2 * np.exp(-np.abs(x))
        else:
            return 1/np.cosh(x)

    def dthetas_update(self, dthetas):
        if self.method == "MinMax":
            #dthetas = np.array([1 if elem >= self.solver.theta_min else 0 for elem in self.solver.v/self.solver.w])
            for i in range(dthetas.shape[0]):
                if self.solver.w[i] < self.near0_eps:
                    dthetas[i] = 0
                else:
                    if self.solver.v[i]/self.solver.w[i] >= self.solver.theta_min:
                        dthetas[i] = 1
                    else:
                        dthetas[i] = 0
        elif self.method == "Smoothing":
            for i in range(dthetas.shape[0]):
                if np.abs(self.solver.w[i]) > self.near0_eps:
                    #print("max v",np.max(self.solver.v),"max w",np.max(self.solver.w),"np.max(self.v/self.w)", np.max(self.solver.v/self.solver.w), "argmax v/w:", np.argmax(self.solver.v/self.solver.w))
                    dthetas[i] = self.kappa * (1/4) * self.sech((-0.5+self.solver.v[i]/self.solver.w[i])*self.kappa)**2
                else:
                    dthetas[i] = 0

    def theta_choice(self, thetas, i, epsilon=1e-100):
        if self.method == "MinMax":
            if self.solver.dim ==1:
                if np.abs(self.solver.w[i]) > epsilon:
                    thetas[i] = min(max(self.solver.theta_min, np.abs(self.solver.v[i]/self.solver.w[i]) ), 
                                    self.solver.env.params_dict["Theta_max"])
                else:
                    thetas[i] = self.solver.theta_st
            else:
                d = self.solver.dim
                for j in range(d):
                    if np.abs(self.solver.w[i]) > epsilon:
                        thetas[j][i] = min(max(self.solver.theta_min, np.abs(self.solver.v[j][i]/self.solver.w[j][i]) ), 
                                        self.solver.env.params_dict["Theta_max"])
                    else:
                        thetas[j][i] = self.solver.theta_st

        elif self.method == "Smoothing":
            #if self.w[i]==0:  #
            if np.abs(self.solver.w[i])<self.near0_eps:  #
                thetas[i] = self.solver.env.params_dict["Theta_max"]
            else:
                thetas[i] = .75 + .25 * np.tanh(self.kappa * (-.5 + self.solver.v[i]/self.solver.w[i]))
        
        else :
            raise ValueError("Wrong Theta choice method type")
        

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


class Mesh:
    def __init__(self, a, b, Nx, tf, t, t_type, boundary, mesh_type="offset_constantDx"):
        self.a = a
        self.b = b
        self.Nx = Nx

        if t_type=="Nt":
            self.Nt = t
        elif t_type=="cfl":
            self.cfl = t
        self.type = mesh_type

        if self.type == "offset_constantDx":
            self.dx = (self.b - self.a)/(self.Nx)
            if t_type=="Nt":
                self.dt = tf / self.Nt
                self.cfl = self.dt/self.dx  #lacks the speed parameter alpha
            elif t_type=="cfl":
                self.dt = self.cfl*self.dx
            self.nodes, self.interfaces = self.grid_offset()

        else:
            raise ValueError("Wrong mesh type")
    
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
            raise ValueError("alpha<0 not available yet")

    def Dx_PtR(self, mesh, b):
        if b == "periodic":
            ret = sparse.diags([np.ones(mesh.Nx+1),-np.ones(mesh.Nx)],
                            [0,-1], shape=(mesh.Nx+1,mesh.Nx+1), format="lil")
            ret[0,-1] = -1
            
        elif b == "dirichlet":
            dia = np.ones(mesh.Nx+1)
            dia[0] = 0
            ret = sparse.diags([dia,-np.ones(mesh.Nx)],
                            [0,-1], shape=(mesh.Nx+1,mesh.Nx+1), format="lil")
        
        return sparse.csr_matrix(ret/mesh.dx)


    def Iter_Mat(self, mesh, theta, alpha, adaptive, flux, boundary):
        if flux=="Upwind":
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
        
        if flux=="LF":
            A = np.zeros(shape=(mesh.Nx+1, mesh.Nx+1))
            for i in range(1,mesh.Nx):
                A[i,i] = 1 + alpha * theta[i] * mesh.dt / mesh.dx
                A[i, i-1] = - alpha * theta[i] * mesh.dt / (2*mesh.dx)
                A[i, i+1] = - alpha * theta[i] * mesh.dt / (2*mesh.dx)

            if boundary=="dirichlet":
                A[0,0] = 1.
                A[-1,-1] = 1.
            elif boundary=="periodic":
                A[0,0] = 1 + alpha * theta[i] * mesh.dt / mesh.dx
                A[0,1] = - alpha * theta[i] * mesh.dt / (2*mesh.dx)
                A[0,-1] = - alpha * theta[i] * mesh.dt / (2*mesh.dx)
                A[-1,0] = - alpha * theta[i] * mesh.dt / (2*mesh.dx)
                A[-1,-2] = - alpha * theta[i] * mesh.dt / (2*mesh.dx)
                A[-1,-1] = 1 + alpha * theta[i] * mesh.dt / mesh.dx
        return sparse.csr_matrix(A)
        
"""    def Iter_Func(self, mesh, theta, alpha, v):   #Function version of Iter_Mat in order to build a linear operator.
                                               #For SATh, as in the case of the simple theta scheme the matrix is only needed to be built once for all.
        for i in range(1,v.size):
            v[i] = v[i]*(1+ alpha*theta[i]*mesh.dt/mesh.dx) - v[i-1] * alpha*theta[i]*mesh.dt/mesh.dx

        return v"""
    

class Functions():
    def __init__(self, mesh, problem, params, tf, init_type, exact):
        self.problem = problem
        self.type = init_type

        if init_type=="bell":
            self.init_func = self.init_bell
        elif init_type=="jump1":
            self.init_func = self.init_jump1
        elif init_type=="jump2":
            self.init_func = self.init_jump2
        elif init_type=='sine_shock':
            self.init_func = self.init_sine
        elif problem == "RIPA":
            self.init_func = self.init_RIPA
        else:
            raise ValueError("invalid init function type")

        self.init_sol = self.init_func(mesh.nodes, params)
        if exact == True:
            self.exact_sol = self.exact(mesh.nodes, params, tf)

    def init_bell(self, x, param, sigma=0.05): #To make a kind of bell curve -> continuous distribution centered in d0=param
        return np.exp(-0.5*((x-param[0])**2)/sigma**2)

    def init_jump1(self, x, params): #To make a piecewise-constant function with a discontinuity in d0=param (1 before, 0 after)
                                    #not compatible with periodical boundaries, shape:      ____  or  ____
                                    #                                                  ____|              |____
        if len(params)!=3:
            raise ValueError("3 values needed to define the initial function")
        u = np.ones_like(x, dtype=float) * params[1]

        for i in range(u.shape[0]):
            if (x[i]>=params[0]):      
                u[i] = params[2]

        return u
    
    def init_jump2(self, x, params):  #
                                      #shape:     ___
                                      #       ___|   |___
        if len(params)!=5:
            raise ValueError("5 values needed to define the initial function")
        u = np.ones_like(x, dtype=float) * params[2]
        for i in range(u.shape[0]):
            if (x[i]<params[1] and x[i]>=params[0]):
                u[i] = params[3]
            elif x[i]>=params[1] :
                u[i] = params[4]
        return u

    def init_sine(self, x, params=[0.5]):
        #For params[0] = 0.5 and x in [0,2]:
        #interval of sol: [-0.5,1.5], alpha_LF = 1.5, time of shock formation: 1/pi~=0.318
        return params[0] + np.sin(np.pi * x)

    def init_RIPA(self, x, params=[]):  #params=[[5,0,3],[1,0,5]]
        #computational domain: [-1,1], Dirichlet BC
        if self.type == "smooth":
            pass
        
        elif self.type == "flat":
            ret = np.empty(shape=(3,x.shape[0]))
            ret[1] = np.zeros_like(x)
            for i in range(x.shape[0]):
                if x[i] < 0:
                    ret[0][i] = 5
                    ret[2][i] = 3
                else:
                    ret[0][i] = 1
                    ret[2][i] = 5

        elif self.type == "nonflat":
            z = np.empty_like(x)
            for i in range(x.shape[0]):
                if x[i] >= -0.4 and x[i] <= -0.2:
                    z[i] = 2 * (np.cos(10*np.pi*(x[i] + 0.3)) +1)
                elif x[i] >= 0.2 and x[i] <= 0.4:
                    z[i] = .5 * (np.cos(10*np.pi*(x[i] - 0.3)) +1)
                else:
                    z[i] = 0
            ret = np.empty(shape=(3,x.shape[0]))
            ret[1] = np.zeros_like(x)
            for i in range(x.shape[0]):
                if x[i] < 0:
                    ret[0][i] = 5 - z[i]
                    ret[2][i] = 3
                else:
                    ret[0][i] = 1 - z[i]
                    ret[2][i] = 5

        else:
            raise ValueError("Wrong init function type for RIPA")
        
        return ret

    def exact(self, x, params, tf):
        if self.problem=="Linear_advection":
            tf = np.ones_like(x)*tf
            x0 = x-tf
            u0 = self.init_func(x0, params)

        elif self.problem=="Burgers":

            u0 = np.zeros(x.size)
            U = self.init_func(x, params)
            for k in range(x.size):
                    
                if k == x.size-1:
                    k_ = 0
                else:
                    k_ = k+1
                
                UL = U[k]
                UR = U[k_]
                                
                if UL > UR:
                    # Shock case:
                    S = 0.5 * (UL + UR)
                    if S >= 0.:
                        UO = UL
                    else:
                        UO = UR
                else:
                    # Rarefaction case
                    if UL >= 0.:
                        UO = UL
                    else:
                        if UR <= 0.:
                            UO = UR
                        else:
                            UO = 0.
                
                u0[k] = 0.5 * UO * UO

            """
            if self.type == "jump1":    
                xf = params[0] + np.max(self.init_sol) * tf
                j1, j2 = 0, 0
                while x[j1] < params[0]:
                    j1 += 1
                    j2 += 1
                while x[j2] < xf:
                    j2 += 1
                u0 = np.ones_like(x) * params[1]
                u0[j1:j2] = ((params[2]-params[1])/(x[j2]-x[j1])) * x[j1:j2] - ((params[2]-params[1])/(x[j2]-x[j1]))*x[j1] + params[1]
                u0[j2:] = params[2]

            elif self.type == "jump2":
                xf1 = params[0] + np.max(self.init_sol) * tf
                xf2 = params[1] + np.max(self.init_sol) * tf /2
                #if xf1 >= xf2 we have rarefaction -> TO DO : split the cases
                #following is the case xf1 < xf2:
                j = 0
                while x[j] < params[0]:
                    j+=1
                j1 = j
                while x[j] < xf1:
                    j += 1
                j2 = j
                while x[j] < xf2:
                    j += 1
                j3 = j
                u0 = np.ones_like(x) * params[2]
                u0[j1:j2] = ((params[3]-params[2])/(x[j2]-x[j1])) * x[j1:j2] - ((params[3]-params[2])/(x[j2]-x[j1]))*x[j1] + params[2]
                u0[j2:j3] = params[3]
                u0[j3:] = params[4]"""

        elif self.problem=="RIPA":
            pass

        return u0


class Theta_Scheme:
    def __init__(self,env):
        self.env = env

    def solver(self):
        t = 0
        u = self.env.funcs.init_sol.copy()
        coef = self.env.mesh.dt*(1-self.env.theta)
        A = self.env.mats.Iter_Mat(self.env.mesh, self.env.theta, self.env.alpha, adaptive=False, flux="Upwind", boundary=None)#

        while (t<self.env.tf):
            t += self.env.mesh.dt
            b = u - coef*self.env.alpha*(self.env.mats.Dx @ u)
            u, _ = sparse.linalg.gmres(A, b)

        return u