# %% [markdown]
# # Simple Unicycle trajectory optimization using SCOCP (no obstacles)

# %%
import cvxpy as cp
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import sympy as sp

import os
import sys
sys.path.append(os.path.join(".."))

import scocp

# %%
np.set_printoptions(precision=4)

# %% [markdown]
# ## Simple unicycle dynamics (no obstacles)

# %%
def get_unicycle_dynamics():
    # construct dynamics
    n_x, n_u = 3, 2
    _x = sp.IndexedBase('x')
    _u = sp.IndexedBase('u')
    x = sp.Matrix([_x[i] for i in range(n_x)])
    u = sp.Matrix([_u[i] for i in range(n_u)])

    # unicycle dynamics
    f_sym = sp.Matrix([
        u[0] * sp.cos(x[2]),  # x_dot = v * cos(theta)
        u[0] * sp.sin(x[2]),  # y_dot = v * sin(theta)
        u[1]                  # theta_dot = omega
    ])

    # jacobian
    jac_x_sym = f_sym.jacobian(x)
    jac_u_sym = f_sym.jacobian(u)

    # create numerical functions
    eval_dynamics = sp.lambdify([x,u], f_sym, modules = [{'atan2':np.arctan2}, 'numpy'])
    eval_dfdx = sp.lambdify([x,u], jac_x_sym, modules = [{'atan2':np.arctan2}, 'numpy'])
    eval_dfdu = sp.lambdify([x,u], jac_u_sym, modules = [{'atan2':np.arctan2}, 'numpy'])

    scipy_rhs_eom = lambda t, x, u: eval_dynamics(x, u).flatten()
    
    def scipy_rhs_eom_augmented(t, x, u):
        assert len(x) == n_x + n_x*n_x + n_x*n_u, f"x must be of length {n_x + n_x*n_x + n_x*n_u}, but got {len(x)}"
        dxdt_aug = np.zeros(n_x + n_x*n_x + n_x*n_u)
        dxdt_aug[:n_x] = scipy_rhs_eom(t, x[:n_x], u)
        A = eval_dfdx(x[:n_x], u)
        B = eval_dfdu(x[:n_x], u)
        dxdt_aug[n_x:n_x*(n_x+1)] = (A @ x[n_x:n_x*(n_x+1)].reshape(n_x,n_x)).flatten()
        dxdt_aug[n_x*(n_x+1):]    = (np.dot(A, x[n_x*(n_x+1):].reshape(n_x,n_u)) + B).flatten()
        return dxdt_aug
    
    return scipy_rhs_eom, scipy_rhs_eom_augmented

# %%
scipy_rhs_eom, scipy_rhs_eom_augmented = get_unicycle_dynamics()

# Build integrator object
integrator = scocp.ScipyIntegrator(
    nx = 3,
    nu = 2,
    rhs = scipy_rhs_eom,
    rhs_stm = scipy_rhs_eom_augmented,
    impulsive = False,
    args = ([0.0, 0.0],),         # place holder for control (length must match nu=2)
    method = 'RK23',
    reltol = 1e-6,
    abstol = 1e-6
)

# %%
# Problem parameters (very simple)
x_initial = np.array([0.0, 0.0, 0.0])           # Start at origin
x_target = np.array([2.0, 2.0, np.pi/4])       # Simple target

# Cost weights  
Q = np.diag([1.0, 1.0, 0.1])      # State cost weights
R = np.diag([0.1, 0.1])           # Control cost weights  
Qf = np.diag([10.0, 10.0, 1.0])   # Terminal cost weights

# Control constraints
v_max = 2.0        # Maximum linear velocity
omega_max = np.pi  # Maximum angular velocity

print(f"Target: {x_target}")

# %%
class SimpleUnicycleProblem(scocp.ContinuousControlSCOCP):
    def __init__(self, integrator, times, trust_region_radius, Q, R, Qf, x_target):
        super().__init__(integrator, times, trust_region_radius=trust_region_radius)
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.x_target = x_target
        return
    
    def evaluate_objective(self, xs, us, vs, ys=None):
        """Quadratic objective function"""
        N, nx = xs.shape
        Nseg, nu = us.shape
        
        # Running cost
        cost = 0.0
        for k in range(Nseg):
            state_error = xs[k,:] - self.x_target
            cost += 0.5 * state_error.T @ self.Q @ state_error
            cost += 0.5 * us[k,:].T @ self.R @ us[k,:]
        
        # Terminal cost
        terminal_error = xs[-1,:] - self.x_target
        cost += 0.5 * terminal_error.T @ self.Qf @ terminal_error
        
        return cost

    def solve_convex_problem(self, xbar, ubar, vbar=None, ybar=None):
        """Solve the convex subproblem (no obstacles)"""
        N, nx = xbar.shape
        _, nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((Nseg, nu), name='control')
        vs = cp.Variable((Nseg, 1), name='v_dummy')  # Dummy variables to avoid None issues
        xis_dyn = cp.Variable((Nseg, nx), name='xi_dyn')         # slack for dynamics
        
        penalty = scocp.get_augmented_lagrangian_penalty(self.weight, xis_dyn, self.lmb_dynamics)
        
        # Quadratic objective
        objective_func = penalty
        for k in range(Nseg):
            state_error = xs[k,:] - self.x_target
            objective_func += 0.5 * cp.quad_form(state_error, self.Q)
            objective_func += 0.5 * cp.quad_form(us[k,:], self.R)
        
        # Terminal cost
        terminal_error = xs[-1,:] - self.x_target
        objective_func += 0.5 * cp.quad_form(terminal_error, self.Qf)

        # Dynamics constraints
        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,:] @ us[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
            for i in range(Nseg)
        ]

        # Trust region constraints
        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <= self.trust_region_radius for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius for i in range(N)
        ]

        # Initial condition
        constraints_initial = [xs[0,:] == x_initial]
        
        # Control bounds
        constraints_control = [
            us[i,0] <= v_max for i in range(Nseg)
        ] + [
            us[i,0] >= -v_max for i in range(Nseg)
        ] + [
            us[i,1] <= omega_max for i in range(Nseg)
        ] + [
            us[i,1] >= -omega_max for i in range(Nseg)
        ]
        
        # Fix dummy v variables to zero (we don't use them in our problem)
        constraints_v_dummy = [vs[i,0] == 0.0 for i in range(Nseg)]

        all_constraints = (constraints_dyn + constraints_trustregion + 
                          constraints_initial + constraints_control + constraints_v_dummy)

        convex_problem = cp.Problem(cp.Minimize(objective_func), all_constraints)
        
        convex_problem.solve(solver=self.solver, verbose=self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, vs.value, None, xis_dyn.value, None, None

# %%
# Create discretized time grid
N = 20  # Very small problem
tf = 4.0  # Short time
times = np.linspace(0, tf, N)
trust_region_radius = 5.0  # Large trust region

# Create optimal control problem object
problem = SimpleUnicycleProblem(integrator, times, trust_region_radius, Q, R, Qf, x_target)

# Create simple initial guess
xs_initial_guess = np.zeros((N, 3))
for i in range(N):
    t = i / (N-1)
    xs_initial_guess[i,:] = x_initial + t * (x_target - x_initial)

us_initial_guess = np.zeros((N-1, 2))
us_initial_guess[:,0] = 0.5  # constant forward velocity
us_initial_guess[:,1] = 0.1  # small angular velocity

# %% 
# Plot initial guess
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(xs_initial_guess[:,0], xs_initial_guess[:,1], 'g--', linewidth=2, label='Initial Guess')
ax.plot(x_initial[0], x_initial[1], 'go', markersize=12, label='Start')
ax.plot(x_target[0], x_target[1], 'ro', markersize=12, label='Target')
ax.set(xlabel="x", ylabel="y")
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("Simple Initial Guess (No Obstacles)")
plt.tight_layout()
plt.show()

# %%
# Solve the problem
problem.verbose_solver = False
tol_feas = 1e-3  # Very relaxed
tol_opt = 1e-2   # Very relaxed
algo = scocp.SCvxStar(problem, tol_opt=tol_opt, tol_feas=tol_feas)

vs_initial_guess = np.zeros((N-1, 1))

try:
    solution = algo.solve(
        xs_initial_guess,
        us_initial_guess,
        vs_initial_guess,
        maxiter=20,
        verbose=True
    )
    print("SUCCESS: Simple unicycle optimization worked!")
    
except Exception as e:
    print(f"ERROR: Even simple optimization failed: {e}")
    solution = None

# %%
# Plot results if successful
if solution is not None and solution.x is not None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # States
    state_labels = ["x", "y", "theta"]
    colors = cm.winter(np.linspace(0, 1, 3))
    for ix in range(3):
        axs[0].plot(problem.times, solution.x[:,ix], label=state_labels[ix], color=colors[ix])
    axs[0].legend()
    axs[0].set(xlabel="Time", ylabel="State")
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title("State Evolution")
    
    # Controls
    control_labels = ["v", "omega"]
    colors = cm.viridis(np.linspace(0, 1, 2))
    axs[1].step(problem.times[:-1], solution.u[:,0], label=control_labels[0], where='post', color=colors[0])
    axs[1].step(problem.times[:-1], solution.u[:,1], label=control_labels[1], where='post', color=colors[1])
    axs[1].legend()
    axs[1].set(xlabel="Time", ylabel="Control")
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title("Control Evolution")
    
    # Trajectory
    axs[2].plot(solution.x[:,0], solution.x[:,1], 'b-', linewidth=2, label='Optimized')
    axs[2].plot(xs_initial_guess[:,0], xs_initial_guess[:,1], 'g--', alpha=0.7, label='Initial Guess')
    axs[2].plot(x_initial[0], x_initial[1], 'go', markersize=10, label='Start')
    axs[2].plot(x_target[0], x_target[1], 'ro', markersize=10, label='Target')
    axs[2].set(xlabel="x", ylabel="y")
    axs[2].set_aspect('equal')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    axs[2].set_title("2D Trajectory")
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("\n" + "="*50)
    print("SIMPLE UNICYCLE SOLUTION SUMMARY")
    print("="*50)
    print(f"Final state:   {solution.x[-1,:]}")
    print(f"Target state:  {x_target}")
    print(f"Final error:   {np.linalg.norm(solution.x[-1,:] - x_target):.6f}")
    
else:
    print("Simple optimization failed - there may be a fundamental issue with the setup")

# %% 