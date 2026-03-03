import dolfin as df
import numpy as np
import math

# Proposed virtual fields
def VFM(u, V1,  alpha1, alpha2, alpha3, T) :
    A_VFM= np.zeros((4, 4))
    B_VFM= np.zeros(4)
    mesh= V1.mesh()
    V = V1.ufl_element()
    tol=1e-10
    dx = df.Measure("dx", domain=mesh)
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    #Example (a)
    class TopBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[1], 1,tol) and on_boundary
    top= TopBoundary()
    top.mark(boundaries, 1)

    """
    Example (b)
    class LeftBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[0], 0,tol) and on_boundary
    left= LeftBoundary()
    left.mark(boundaries, 1) """

    def epsilon(u):
        return 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T) 

    ##### VF(1) 
    L_V = df.TrialFunction(V1)
    w = df.TestFunction(V1)
    a1 = df.inner(df.nabla_grad(L_V), df.nabla_grad(w)) * dx   
    l1 = epsilon(u)[0,0]* epsilon(w)[0,0] * dx

    def custom_boundary(x, on_boundary):
        return on_boundary  and ( x[1]<tol)
    bc = df.DirichletBC(V1, df.Constant((0,0)), custom_boundary)
    bc_LV = [bc]
    L_V = df.Function(V1)

    df.solve(a1 == l1, L_V, bc_LV, solver_parameters={'linear_solver': 'mumps'})
    R = df.FiniteElement('R', mesh.ufl_cell(), degree=0)
    r_elements = [R for _ in range(3)] 

    mixed_r_elements = df.MixedElement(r_elements)  
    R1 = df.FunctionSpace(mesh, mixed_r_elements) 
    W1_element  = df.MixedElement([V, mixed_r_elements])
    W1 = df.FunctionSpace(mesh, W1_element)

    (L_H, m ) = df.TrialFunctions(W1)
    (w , n  ) = df.TestFunctions(W1)
    
    penalty_term1 = alpha1* m[0]* (epsilon(u)[1,1]* epsilon(w)[0,0]+epsilon(u)[0,0]* epsilon(w)[1,1] ) * dx +  n[0]*(epsilon(u)[1,1]* epsilon(L_H)[0,0]+epsilon(u)[0,0]* epsilon(L_H)[1,1] ) * dx - m[0]*n[0] *dx  
    penalty_term2 = alpha2* m[1]*epsilon(u)[1,1]* epsilon(w)[1,1] * dx  + n[1]*epsilon(u)[1,1]* epsilon(L_H)[1,1] * dx - m[1]*n[1] *dx 
    penalty_term3 = alpha3* m[2]*epsilon(u)[0,1]* epsilon(w)[0,1] * dx  +  n[2]*epsilon(u)[0,1]* epsilon(L_H)[0,1] * dx - m[2]*n[2] *dx 
    
    a =  df.inner(df.nabla_grad(L_H), df.nabla_grad(w)) * dx   
    a += penalty_term1 +penalty_term2 +penalty_term3 
    l =  df.inner(df.nabla_grad(L_V), df.nabla_grad(w)) * dx  

    bc_v = df.DirichletBC(W1.sub(0), df.Constant((0,0)), custom_boundary)
    bc_LH = [bc_v]
    s= df.Function(W1)
    df.solve(a == l, s, bc_LH)
    (L_H, m) = s.split()

    norm_LH= math.sqrt( df.assemble(df.inner(df.nabla_grad(L_H), df.nabla_grad(L_H))*dx) )
    L_H = df.project(L_H/ norm_LH, V1)

    m_vec = df.interpolate(m,R1).vector().get_local()   
    print("Values of m :", m_vec)

    A_VFM[0][0]= df.assemble(epsilon(u)[0,0]* epsilon(L_H)[0,0] * dx ) 
    A_VFM[0][1]= df.assemble((epsilon(u)[1,1]* epsilon(L_H)[0,0] * dx +epsilon(u)[0,0]* epsilon(L_H)[1,1] * dx  ))
    A_VFM[0][2]= df.assemble(epsilon(u)[1,1]* epsilon(L_H)[1,1] * dx)
    A_VFM[0][3]= df.assemble(2*epsilon(u)[0,1]* epsilon(L_H)[0,1] * dx )
    B_VFM[0]= df.assemble(df.dot(T,L_H)*ds(1)) 


    ##### VF(2) 
    L_V = df.TrialFunction(V1)
    w = df.TestFunction(V1)
    a1 = df.inner(df.nabla_grad(L_V), df.nabla_grad(w)) * dx   
    l1 =  epsilon(u)[1,1]* epsilon(w)[0,0] * dx +epsilon(u)[0,0]* epsilon(w)[1,1] * dx

    bc = df.DirichletBC(V1, df.Constant((0,0)), custom_boundary)
    bc_LV = [bc]
    L_V = df.Function(V1)

    df.solve(a1 == l1, L_V, bc_LV, solver_parameters={'linear_solver': 'mumps'})

    (L_H, m ) = df.TrialFunctions(W1)
    (w , n  ) = df.TestFunctions(W1)
    
    penalty_term1 = math.sqrt(alpha1)* m[0]*epsilon(u)[0,0]* epsilon(w)[0,0] * dx  +  math.sqrt(alpha2)* n[0]*epsilon(u)[0,0]* epsilon(L_H)[0,0] * dx - m[0]*n[0] *dx 
    penalty_term2 = math.sqrt(alpha2)* m[1]*epsilon(u)[1,1]* epsilon(w)[1,1] * dx  +  math.sqrt(alpha2)* n[1]*epsilon(u)[1,1]* epsilon(L_H)[1,1] * dx - m[1]*n[1] *dx 
    penalty_term3 = math.sqrt(alpha3)* m[2]*epsilon(u)[0,1]* epsilon(w)[0,1] * dx  +  math.sqrt(alpha3)* n[2]*epsilon(u)[0,1]* epsilon(L_H)[0,1] * dx - m[2]*n[2] *dx 

    a =  df.inner(df.nabla_grad(L_H), df.nabla_grad(w)) * dx  
    a += penalty_term1 + penalty_term2 + penalty_term3    

    l =  df.inner(df.nabla_grad(L_V), df.nabla_grad(w)) * dx

    bc_v = df.DirichletBC(W1.sub(0), df.Constant((0,0)), custom_boundary)
    bc_LH = [bc_v]
    s= df.Function(W1)
    df.solve(a == l, s, bc_LH)
    (L_H, m) = s.split()

    norm_LH= math.sqrt( df.assemble(df.inner(df.nabla_grad(L_H), df.nabla_grad(L_H))*dx) )
    L_H = df.project(L_H/ norm_LH, V1)

    m_vec = df.interpolate(m,R1).vector().get_local()  
    print("Values of m :", m_vec)

    A_VFM[1][0]= df.assemble(epsilon(u)[0,0]* epsilon(L_H)[0,0] * dx ) 
    A_VFM[1][1]= df.assemble((epsilon(u)[1,1]* epsilon(L_H)[0,0] * dx +epsilon(u)[0,0]* epsilon(L_H)[1,1] * dx  ))
    A_VFM[1][2]= df.assemble(epsilon(u)[1,1]* epsilon(L_H)[1,1] * dx)
    A_VFM[1][3]= df.assemble(2*epsilon(u)[0,1]* epsilon(L_H)[0,1] * dx ) 
    B_VFM[1]= df.assemble(df.dot(T,L_H)*ds(1))


    ##### VF(3)
    L_V = df.TrialFunction(V1)
    w = df.TestFunction(V1)
    a1 = df.inner(df.nabla_grad(L_V), df.nabla_grad(w)) * dx   
    l1 = epsilon(u)[1,1]* epsilon(w)[1,1] * dx

    bc = df.DirichletBC(V1, df.Constant((0,0)), custom_boundary)
    bc_LV = [bc]
    L_V = df.Function(V1)

    df.solve(a1 == l1, L_V, bc_LV, solver_parameters={'linear_solver': 'mumps'})


    (L_H, m ) = df.TrialFunctions(W1)
    (w , n  ) = df.TestFunctions(W1)
    
    penalty_term1 = math.sqrt(alpha1)* m[0]* (epsilon(u)[1,1]* epsilon(w)[0,0]+epsilon(u)[0,0]* epsilon(w)[1,1] ) * dx +  math.sqrt(alpha1)* n[0]*(epsilon(u)[1,1]* epsilon(L_H)[0,0]+epsilon(u)[0,0]* epsilon(L_H)[1,1] ) * dx - m[0]*n[0] *dx  
    penalty_term2 = math.sqrt(alpha2)* m[1]*epsilon(u)[0,0]* epsilon(w)[0,0] * dx  +  math.sqrt(alpha2)* n[1]*epsilon(u)[0,0]* epsilon(L_H)[0,0] * dx - m[1]*n[1] *dx 
    penalty_term3 = math.sqrt(alpha3)* m[2]*epsilon(u)[0,1]* epsilon(w)[0,1] * dx  +  math.sqrt(alpha3)* n[2]*epsilon(u)[0,1]* epsilon(L_H)[0,1] * dx - m[2]*n[2] *dx 

    a =  df.inner(df.nabla_grad(L_H), df.nabla_grad(w)) * dx   
    a += penalty_term1 +penalty_term2 +penalty_term3 
    l =  df.inner(df.nabla_grad(L_V), df.nabla_grad(w)) * dx  

    bc_v = df.DirichletBC(W1.sub(0), df.Constant((0,0)), custom_boundary)
    bc_LH = [bc_v]
    s= df.Function(W1)
    df.solve(a == l, s, bc_LH)
    (L_H, m) = s.split()

    norm_LH= math.sqrt( df.assemble(df.inner(df.nabla_grad(L_H), df.nabla_grad(L_H))*dx) )
    L_H = df.project(L_H/ norm_LH, V1)

    m_vec = df.interpolate(m,R1).vector().get_local() 
    print("Values of m :", m_vec)

    A_VFM[2][0]= df.assemble(epsilon(u)[0,0]* epsilon(L_H)[0,0] * dx ) 
    A_VFM[2][1]= df.assemble((epsilon(u)[1,1]* epsilon(L_H)[0,0] * dx +epsilon(u)[0,0]* epsilon(L_H)[1,1] * dx  ))
    A_VFM[2][2]= df.assemble(epsilon(u)[1,1]* epsilon(L_H)[1,1] * dx)
    A_VFM[2][3]= df.assemble(2*epsilon(u)[0,1]* epsilon(L_H)[0,1] * dx )
    B_VFM[2]= df.assemble(df.dot(T,L_H)*ds(1))


    ##### VF(4)
    L_V = df.TrialFunction(V1)
    w = df.TestFunction(V1)
    a1 = df.inner(df.nabla_grad(L_V), df.nabla_grad(w)) * dx   
    l1 = epsilon(u)[0,1]* epsilon(w)[0,1] * dx

    bc = df.DirichletBC(V1, df.Constant((0,0)), custom_boundary)
    bc_LV = [bc]
    L_V = df.Function(V1)

    df.solve(a1 == l1, L_V, bc_LV, solver_parameters={'linear_solver': 'mumps'})


    (L_H, m ) = df.TrialFunctions(W1)
    (w , n  ) = df.TestFunctions(W1)
    
    penalty_term1 = math.sqrt(alpha1)* m[0]* (epsilon(u)[1,1]* epsilon(w)[0,0]+epsilon(u)[0,0]* epsilon(w)[1,1] ) * dx +   math.sqrt(alpha1)* n[0]*(epsilon(u)[1,1]* epsilon(L_H)[0,0]+epsilon(u)[0,0]* epsilon(L_H)[1,1] ) * dx - m[0]*n[0] *dx  
    penalty_term2 = math.sqrt(alpha2)* m[1]*epsilon(u)[0,0]* epsilon(w)[0,0] * dx  +  math.sqrt(alpha2)* n[1]*epsilon(u)[0,0]* epsilon(L_H)[0,0] * dx - m[1]*n[1] *dx 
    penalty_term3 = math.sqrt(alpha3)* m[2]*epsilon(u)[1,1]* epsilon(w)[1,1] * dx  + math.sqrt(alpha3)* n[2]*epsilon(u)[1,1]* epsilon(L_H)[1,1] * dx - m[2]*n[2] *dx 
    
    a =  df.inner(df.nabla_grad(L_H), df.nabla_grad(w)) * dx   
    a += penalty_term1 +penalty_term2 +penalty_term3 
    l =  df.inner(df.nabla_grad(L_V), df.nabla_grad(w)) * dx  

    bc_v = df.DirichletBC(W1.sub(0), df.Constant((0,0)), custom_boundary)
    bc_LH = [bc_v]
    s= df.Function(W1)
    df.solve(a == l, s, bc_LH)
    (L_H, m) = s.split()

    norm_LH= math.sqrt( df.assemble(df.inner(df.nabla_grad(L_H), df.nabla_grad(L_H))*dx) )
    L_H = df.project(L_H/ norm_LH, V1)

    m_vec = df.interpolate(m,R1).vector().get_local()   
    print("Values of m :", m_vec)

    A_VFM[3][0]= df.assemble(epsilon(u)[0,0]* epsilon(L_H)[0,0] * dx ) 
    A_VFM[3][1]= df.assemble((epsilon(u)[1,1]* epsilon(L_H)[0,0] * dx +epsilon(u)[0,0]* epsilon(L_H)[1,1] * dx  ))
    A_VFM[3][2]= df.assemble(epsilon(u)[1,1]* epsilon(L_H)[1,1] * dx)
    A_VFM[3][3]= df.assemble(2*epsilon(u)[0,1]* epsilon(L_H)[0,1] * dx ) 
    B_VFM[3]= df.assemble(df.dot(T,L_H)*ds(1))


    X=np.linalg.solve(A_VFM,B_VFM)
    print("X=", X) 

    norm_LH= math.sqrt( df.assemble(df.inner(L_H,L_H)*ds(1)) )

    return A_VFM,B_VFM, X 
  