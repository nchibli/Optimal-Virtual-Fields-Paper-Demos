import dolfin as df
import numpy as np
import ufl
import math

def transversely_isotropic_elasticity_problem(
        mesh_params={},
        mat_params={},
        load_params={}):        
    
    Nx = mesh_params.get("Nx", 100)
    Ny = mesh_params.get("Ny", 100)
    degree = mesh_params.get("degree", 3)
    mesh = df.UnitSquareMesh(Nx, Ny)
    dx = df.Measure("dx", domain=mesh)

    coef_a = mat_params.get("coef_a",)
    coef_b = mat_params.get("coef_b",)
    coef_c = mat_params.get("coef_c",)
    coef_d = mat_params.get("coef_d",)

    u_boundary= load_params.get("u_boundary",)
    T= load_params.get("T",)
    f= load_params.get("f",)

    V = df.VectorElement('CG', mesh.ufl_cell(), degree=degree)
    V1 = df.FunctionSpace(mesh,V)
 
    dx = df.Measure("dx", domain=mesh)

    tol = 1e-15
    def custom_boundary(x, on_boundary):
         return on_boundary and x[1]< tol
    bc_v = df.DirichletBC(V1, u_boundary, custom_boundary)
    bc = [bc_v]
 
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    #Example (a)
    class TopBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[1], 1, tol) and on_boundary
    top= TopBoundary()
    top.mark(boundaries, 1) 

    """
    Example (b)
    class LeftBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[0], 0, tol) and on_boundary
    left= LeftBoundary()
    left.mark(boundaries, 1)"""

    u = df.TrialFunction(V1)
    v = df.TestFunction(V1)

    def epsilon(w):
        return 0.5*(df.nabla_grad(w) + df.nabla_grad(w).T)

    a = ( coef_a*epsilon(u)[0,0]*epsilon(v)[0,0]
    + coef_b*(epsilon(u)[1,1]*epsilon(v)[0,0] + epsilon(u)[0,0]*epsilon(v)[1,1])
    + coef_c*epsilon(u)[1,1]*epsilon(v)[1,1]
    + 2*coef_d*epsilon(u)[0,1]*epsilon(v)[0,1]
    ) * dx
    l = df.dot(T, v) * ds(1) 

    u = df.Function(V1)
    df.solve(a == l,  u, bc)
    u = df.interpolate(u,V1)
    u_norm= math.sqrt(df.assemble(u**2*dx)+ df.assemble((epsilon(u) )**2 * dx ))

    return u, u_norm

 