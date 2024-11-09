# https://www.karlin.mff.cuni.cz/~hron/fenics-tutorial/convection_diffusion/doc.html
# https://stackoverflow.com/questions/51763982/solving-the-heat-equation-by-fenics
#



# https://www.youtube.com/watch?v=QpA7E4YHbyU

import  fenics as fe
import matplotlib.pyplot as plt


n_elements = 32
mesh = fe.UnitIntervalMesh(n_elements)

#define a functional space
lagrange_polynomial_space_1st_order = fe.FunctionSpace(mesh,"Lagrange",1)

#the value of the solution on the boundary
u_on_boundary = fe.Constant(0.0)

# a function to return whether we are on the boundary
def boundary_boolean_function(x,on_boundary):
    return on_boundary

# the homogenous Dirichlet boundary conditions
boundary_condition = fe.DirichletBC(
    lagrange_polynomial_space_1st_order,
    u_on_boundary,
    boundary_boolean_function,
)

# the initial condition, u((t=0,x) = sin(pi*x)
initial_condition = fe.Expression(
             "sin(3.1415*x[0])",
             degree = 1
)

# Discretize the initial condition
u_old = fe.interpolate(
           initial_condition,
           lagrange_polynomial_space_1st_order
)
plt.figure()
fe.plot(u_old,label="t = 0.0")


# The time stepping
time_step_length = 0.1

# the forcing on the rhs
heat_source = fe.Constant(0.0)

# create the finite element problem
u_trial = fe.TrialFunction(lagrange_polynomial_space_1st_order)
v_test  = fe.TestFunction(lagrange_polynomial_space_1st_order)

vel = fe.Constant(1)
# weak form is made from Heat equation with the advection term taken from
# https://fenicsproject.org/qa/2828/implementing-a-very-simple-1d-advection-diffusion-demo/
weak_form_residuum = (
    u_trial * v_test *fe.dx
  + time_step_length * fe.div(vel*fe.grad(u_trial))*v_test*fe.dx
-
(
      u_old *v_test * fe.dx
    + time_step_length*heat_source *v_test*fe.dx
)
)

weak_form_lhs = fe.lhs(weak_form_residuum)
weak_form_rhs = fe.rhs(weak_form_residuum)

u_solution = fe.Function(lagrange_polynomial_space_1st_order)

n_time_steps = 5

time_current = 0.0

for i in range(n_time_steps):
    time_current += time_step_length

    # Finite Element assembly, BC imprint & solving the linear system
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        boundary_condition
    )
    u_old.assign(u_solution)
    fe.plot(u_solution,label=f"t={time_current:1.1f}")
    plt.show()
    plt.savefig('heat_time.png')
    qq = 0

