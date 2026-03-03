import dolfin as df
import numpy as np

# Proposed virtual fields
def VFM(phi, u, V_mu, V, Q, alpha, rho, omega) :

    mesh= V_mu.mesh()
    dx = df.Measure("dx", domain=mesh)
    def epsilon(u):
        return 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T) 
    tol = 1e-10
    V1 = df.FunctionSpace(mesh, V)

 
    mu_rec = df.Function(V_mu)
    mu_rec.vector() [:]=1
    n_phi= len(phi)
    coefficients= []
    A_VFM= np.zeros((n_phi, n_phi))
    B_VFM= np.zeros(n_phi)

    for i in range(n_phi):
        def custom_boundary1(x, on_boundary):
          return on_boundary and x[1]<tol
        def custom_boundary2(x, on_boundary):
          return on_boundary   and x[1]>1-tol          

        if n_phi>=2:
           penalty_term = 0
        
           R = df.FiniteElement('R', mesh.ufl_cell(), degree=0)
           r_elements = [R for _ in range(n_phi - 1)]

           mixed_r_elements = df.MixedElement(r_elements)  
           R1 = df.FunctionSpace(mesh, mixed_r_elements) 
           W1_element  = df.MixedElement([V, Q, mixed_r_elements])
           W1 = df.FunctionSpace(mesh, W1_element)

           (L_H, p, m ) = df.TrialFunctions(W1)
           (w, q, n  ) = df.TestFunctions(W1)
           index=0

           for j in range(n_phi):
             if i != j: 
               mj = m[index]
               nj = n[index]
               penalty_term  +=  df.sqrt(alpha)*df.inner(mj*phi[j]*epsilon(u), epsilon(w)) *dx   + df.sqrt(alpha)*df.inner(nj*phi[j]*epsilon(u), epsilon(L_H)) *dx   - mj*nj *dx 
               index += 1

           a =   df.inner(df.nabla_grad(L_H), df.nabla_grad(w)) * dx + p*df.div(w)*dx  + df.div(L_H) * q * dx 
           a += penalty_term    

           l =  df.inner(phi[i] * epsilon(u), epsilon(w)) * dx # 

           bc1 = df.DirichletBC(W1.sub(0), df.Constant((0, 0)), custom_boundary1)
           bc2 = df.DirichletBC(W1.sub(0).sub(1), df.Constant(0), custom_boundary2)
           bc_LH = [bc1, bc2]

           s= df.Function(W1)
           df.solve(a == l, s, bc_LH, solver_parameters={'linear_solver': 'mumps'})
        
           (L_H, p, m) = s.split()

           for j in range(n_phi):
              A_VFM[i][j]= df.assemble(df.inner(phi[j]*epsilon(u), epsilon(L_H)) *dx)
            
           B_VFM[i]= 1/2*( rho*omega**2*df.assemble(df.inner(u,L_H)*dx)- 2*df.assemble(df.inner(epsilon(u), epsilon(L_H))*dx))
           m_vec = df.interpolate(m,R1).vector().get_local() 
           print("Values of m:", m_vec)

        else:
           W1_element  = df.MixedElement([V, Q])
           W1 = df.FunctionSpace(mesh, W1_element)

           (L_H, p) = df.TrialFunctions(W1)
           (w, q   ) = df.TestFunctions(W1)
           a =   df.inner(df.nabla_grad(L_H), df.nabla_grad(w)) * dx + p*df.div(w)*dx + df.div(L_H) * q * dx

           l =  df.inner(phi[i] * epsilon(u), epsilon(w)) * dx  
           bc1 = df.DirichletBC(W1.sub(0), df.Constant((0, 0)), custom_boundary1)
           bc2 = df.DirichletBC(W1.sub(0).sub(1), df.Constant(0), custom_boundary2)
           bc_LH = [bc1, bc2]

           s= df.Function(W1)
           df.solve(a == l, s, bc_LH, solver_parameters={'linear_solver': 'mumps'})
        
           (L_H, p) = s.split()
           for j in range(n_phi):
              A_VFM[i][j]= df.assemble(df.inner(phi[j]*epsilon(u), epsilon(L_H)) *dx)
        
           B_VFM[i]= 1/2*( rho*omega**2*df.assemble(df.inner(u,L_H)*dx) - 2*df.assemble(df.inner(epsilon(u), epsilon(L_H))*dx))
 
    print(A_VFM)
    coefficients = np.linalg.solve(A_VFM,B_VFM) 
    for i in range(n_phi):
       mu_rec += coefficients[i] * phi[i]
       print( coefficients[i] )
    mu_rec = df.project(mu_rec ,V_mu) 
    return mu_rec


def solve_VFM_2_dim(v1,v2,phi_list, u1,u2, rho, omega, V_mu):
    mesh= V_mu.mesh()
    dx = df.Measure("dx", domain=mesh) 

    def epsilon(u):
        return 0.5*(df.nabla_grad(u) + df.nabla_grad(u).T) 
    
    A_VFM= np.zeros((2,2))
    B_VFM= np.zeros(2)
    A_VFM[0][0]= df.assemble(df.inner(phi_list[0]*epsilon(u1), epsilon(v1)) *dx)      
    A_VFM[0][1]= df.assemble(df.inner(phi_list[1]*epsilon(u1), epsilon(v1)) *dx)
    A_VFM[1][0]= df.assemble(df.inner(phi_list[0]*epsilon(u2), epsilon(v2)) *dx)
    A_VFM[1][1]= df.assemble(df.inner(phi_list[1]*epsilon(u2), epsilon(v2)) *dx)

    B_VFM[0]= 1/2*( (rho*omega**2*df.assemble(df.inner(u1,v1)*dx) - 2*df.assemble(df.inner(epsilon(u1), epsilon(v1))*dx) ) )
    B_VFM[1]= 1/2*( (rho*omega**2*df.assemble(df.inner(u2,v2)*dx) - 2*df.assemble(df.inner(epsilon(u2), epsilon(v2))*dx) ) )
    
    coefficients=np.linalg.solve(A_VFM,B_VFM)
    mu_rec = df.Function(V_mu)
    mu_rec.vector() [:]=1
    
    for i in range(2):
       mu_rec += coefficients[i] * phi_list[i]
       print( coefficients[i] )
    mu_rec = df.project(mu_rec ,V_mu) 
    
    return mu_rec
 



