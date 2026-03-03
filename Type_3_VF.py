import dolfin as df
import numpy as np
import ufl
import math

def type_3(
        mesh_params={},
        Vertical_traction = True):        
    
    Nx = mesh_params.get("Nx", 100)
    Ny = mesh_params.get("Ny", 100)
    degree = mesh_params.get("degree", 3)
    f = df.Expression(('x[0]*cos(10*(x[0]+x[1]))', 'x[0]*sin(10*(x[0]+x[1]))'), degree=3)

    mesh = df.UnitSquareMesh(Nx, Ny)
    dx = df.Measure("dx", domain=mesh)
    tol = 1e-15

    V = df.VectorElement('CG', mesh.ufl_cell(), degree=degree)
    V1 = df.FunctionSpace(mesh,V)
    Q = df.FiniteElement('CG', mesh.ufl_cell(), degree=degree-1)

    W_element = df.MixedElement([V, Q])
    W = df.FunctionSpace(mesh, W_element)
 
    dx = df.Measure("dx", domain=mesh)

    def custom_boundary(x, on_boundary):
         return on_boundary and x[1]< tol
    bc1 = df.DirichletBC(W.sub(0), df.Constant((0,0)), custom_boundary)
    def custom_boundary(x, on_boundary):
        return on_boundary and x[1]>1-tol  
    
    if Vertical_traction == True :
        bc2 = df.DirichletBC(W.sub(0).sub(1), df.Constant(0), custom_boundary) 
    else: 
        bc2 = df.DirichletBC(W.sub(0).sub(0), df.Constant(0), custom_boundary) 

    bc = [bc1, bc2]

    (u, p) = df.TrialFunctions(W)
    (v, q) = df.TestFunctions(W)
 
    def epsilon(u):
        return 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T) 

    a =  df.inner(epsilon(u), epsilon(v)) * dx + p* ufl.nabla_div(v)*dx- ufl.nabla_div(u)*q*dx 
    l = df.dot(f,v) * dx  
    
    w = df.Function(W)
    df.solve(a == l, w, bc, solver_parameters={'linear_solver':'mumps'})
    u = df.Function(V1)
    (u,p) = w.split() 
    u = df.interpolate(u,V1)
 
    return u

 