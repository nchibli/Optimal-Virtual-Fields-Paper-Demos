import dolfin as df
import numpy as np
import ufl
import math

def isotropic_elasticity_problem(
        mesh_params={},
        mat_params={},
        load_params={}):        
    
    Nx = mesh_params.get("Nx", 100)
    Ny = mesh_params.get("Ny", 100)
    degree = mesh_params.get("degree", 3)
    mu = mat_params.get("mu", 2)
    rho = mat_params.get("rho", 1)
    omega = mat_params.get("omega", 1)
    u_boundary= load_params.get("u_boundary",)
    T= load_params.get("T",)
    f= load_params.get("f",)
    mesh = df.UnitSquareMesh(Nx, Ny)
    dx = df.Measure("dx", domain=mesh)

    V = df.VectorElement('CG', mesh.ufl_cell(), degree=degree)
    V1 = df.FunctionSpace(mesh,V)
    Q = df.FiniteElement('CG', mesh.ufl_cell(), degree=degree-1)

    f = df.Constant((0,0))
    W_element = df.MixedElement([V, Q])
    W = df.FunctionSpace(mesh, W_element)
 
    dx = df.Measure("dx", domain=mesh)

    tol = 1e-15
    def custom_boundary(x, on_boundary):
         return on_boundary and x[1]< tol
    bc_v = df.DirichletBC(W.sub(0), u_boundary, custom_boundary)
    bc = [bc_v]
 
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    class TopBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[1], 1, tol) and on_boundary
    top= TopBoundary()
    top.mark(boundaries, 1)

    (u, p) = df.TrialFunctions(W)
    (v, q) = df.TestFunctions(W)
 
    def epsilon(u):
        return 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T) 

    # Define stress tensor
    def sigma(u, p):
        d = u.geometric_dimension() 
        return 2*mu*epsilon(u) + p*df.Identity(d)

    # Define variational problem
    a = df.inner(sigma(u, p), epsilon(v)) * dx - df.inner(rho*omega**2*u,v)*dx - ufl.nabla_div(u)*q*dx 
    l = df.dot(f,v) * dx + df.dot(T ,v)*ds(1)

    # Compute solution
    w = df.Function(W)
    df.solve(a == l, w, bc, solver_parameters={'linear_solver':'mumps'})
    u = df.Function(V1)
    (u,p) = w.split() 
    u = df.interpolate(u,V1)
    u_norm = math.sqrt(df.assemble(u**2*dx)+ df.assemble((epsilon(u) )**2 * dx ))

    V_mu = df.FunctionSpace(mesh, 'CG', degree)  
    mu_function = df.interpolate(mu, V_mu)
 
    return u, mu_function, u_norm

 