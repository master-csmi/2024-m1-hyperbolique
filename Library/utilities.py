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

    def theta_choice(self, thetas, i, epsilon=1e-6):
        if self.method == "MinMax":
            if np.abs(self.solver.w[i]) > epsilon:
                thetas[i] = min(max(self.solver.theta_min, np.abs(self.solver.v[i]/self.solver.w[i]) ), 1)
            else:
                thetas[i] = self.solver.theta_st
        
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
        if boundary == "constant":  #Neumann at the left
            self.Nx += 1
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
        
    def Iter_Func(self, mesh, theta, alpha, v):   #Function version of Iter_Mat in order to build a linear operator.
                                               #For SATh, as in the case of the simple theta scheme the matrix is only needed to be built once for all.
        for i in range(1,v.size):
            v[i] = v[i]*(1+ alpha*theta[i]*mesh.dt/mesh.dx) - v[i-1] * alpha*theta[i]*mesh.dt/mesh.dx

        return v
    

class Functions():
    def __init__(self, mesh, problem, params, tf, init_type):
        self.problem = problem

        if init_type=="bell":
            self.init_func = self.init_bell
        elif init_type=="jump1":
            self.init_func = self.init_jump1
        elif init_type=="jump2":
            self.init_func = self.init_jump2
        else:
            raise ValueError("invalid init function type")

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
            raise ValueError("2 values needed for the coordinates of the perturbation")
        u = np.zeros_like(x, dtype=float)
        for i in range(u.shape[0]):
            if (x[i]<params[1] and x[i]>=params[0]):
                u[i] = 1
        return u

    def exact(self, x, params, tf):
        tf = np.ones_like(x)*tf
        if self.problem=="Linear_advection":
            x0 = x-tf
            u0 = self.init_func(x0, params)
        elif self.problem=="Burgers":
            u0 = np.zeros_like(x)
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
        A = self.env.mats.Iter_Mat(self.env.mesh, self.env.theta, self.env.alpha, adaptive=False)

        while (t<self.env.tf):
            t += self.env.mesh.dt
            b = u - coef*self.env.alpha*(self.env.mats.Dx @ u)
            u, _ = sparse.linalg.gmres(A, b)

        return u