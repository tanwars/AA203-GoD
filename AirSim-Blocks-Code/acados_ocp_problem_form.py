from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from scipy import linalg as la

def create_ocp_solver(model, Tf, dt, kq, kr, n, m, 
                    max_abs_roll_rate, max_abs_pitch_rate,
                    max_abs_yaw_rate, x0):
    '''
    creates ACADOS ocp solver with conditions and costs 
    '''

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    N = int(Tf/dt)
    ocp.dims.N = N

    # set cost
    Q = kq * np.eye(n)
    R = kr * np.eye(m)

    ocp.cost.W_e = Q
    ocp.cost.W = la.block_diag(Q, R)

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx:,:] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    # solver options

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
    ocp.solver_options.tf = Tf

    # constraints on control
    ocp.constraints.lbu = np.array([-max_abs_roll_rate, -max_abs_pitch_rate, -max_abs_pitch_rate,0])
    ocp.constraints.ubu = np.array([max_abs_roll_rate, max_abs_pitch_rate, max_abs_pitch_rate,1])
    ocp.constraints.idxbu = np.array([0,1,2,3])
    ocp.constraints.x0 = x0

    ocp.cost.yref  = np.zeros((n+m,))
    ocp.cost.yref_e = np.zeros((n,))

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

    return ocp_solver

