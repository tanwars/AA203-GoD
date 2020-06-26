import numpy as np

def linear_quad_model(num_states, g = 9.81):
    ## linearized about x = 0 , u = [0,0,0,g]
    A = np.zeros((num_states,num_states))
    B = np.zeros((num_states,4))

    A[0,6] = 1
    A[1,7] = 1
    A[2,8] = 1
    A[6,4] = -g
    A[7,3] = g
    
    B[3,0] = 1
    B[4,1] = 1
    B[5,2] = 1
    B[8,3] = -1

    return A, B

from acados_template import AcadosModel
from casadi import SX, MX, vertcat, sin, cos, mtimes, Function, inv, jacobian

def acados_linear_quad_model(num_states, g = 9.81):

    model_name = 'linear_quad'

    A, B = linear_quad_model(num_states)
    tc = 0.59375 # mapping control to throttle
    # tc = 0.75
    scaling_control_mat = np.array([1, 1, 1, (g/tc)])
    f_const = 20
    # scaling_control_mat = np.array([1, 1, 1, ((g + f_const)/tc)])
    # add_mat = np.array([0,0,0,-f_const])
    # scaling_control_mat = np.array([1,1,1,1])

    # set up states & controls
    x = SX.sym('x',9)
    u = SX.sym('u',4)

    # xdot
    xdot = SX.sym('xdot',9)

    #parameters
    p = []
    
    f_expl = mtimes(A,x) + mtimes(B,u * scaling_control_mat)
    # f_expl = mtimes(A,x) + mtimes(B,u * scaling_control_mat + add_mat)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model

def acados_linear_quad_model_moving_eq(num_states, g = 9.81):

    model_name = 'linear_quad_v2'
    tc = 0.59375
    # scaling_control_mat = np.array([1, 1, 1, (g/tc)])
    gravity = SX.zeros(3)
    gravity[2] = g

    x = SX.sym('x',9)
    u = SX.sym('u',4)
    xdot = SX.sym('xdot',9)

    r1 = SX(3,3)
    r1[0,0] = 1
    r1[1,1] = cos(x[3])
    r1[1,2] = -sin(x[3])
    r1[2,1] = sin(x[3])
    r1[2,2] = cos(x[3])

    r2 = SX(3,3)
    r2[0,0] = cos(x[4])
    r2[0,2] = sin(x[4])
    r2[1,1] = 1
    r2[2,0] = -sin(x[4])
    r2[2,2] = cos(x[4])

    r3 = SX(3,3)
    r3[0,0] = cos(x[5])
    r3[0,1] = -sin(x[5])
    r3[1,0] = sin(x[5])
    r3[1,1] = cos(x[5])
    r3[2,2] = 1

    tau = SX(3,3)
    tau[:,0] = (inv(r2))[:,0]
    tau[:,1] = (inv(r2))[:,1]
    tau[:,2] = (inv(r2 @ r1))[:,2]

    f_nl = vertcat(
        x[6:9],
        inv(tau) @ u[0:3], 
        - ((r3 @ r1 @ r2)[:,2]) * u[3] * (g/tc) + gravity
    )

    # f_const = 20
    # f_nl = vertcat(
    #     x[6:9],
    #     inv(tau) @ u[0:3], 
    #     - ((r3 @ r1 @ r2)[:,2]) * (u[3] * ((g + f_const)/tc) - f_const) + gravity
    # )

    A = jacobian(f_nl, x)
    B = jacobian(f_nl, u)

    p = []
    
    f_expl = mtimes(A,x) + mtimes(B,u)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model


def acados_nonlinear_quad_model(num_states, g = 9.81):

    model_name = 'nonlinear_quad'

    position = SX.sym('o',3)
    velocity = SX.sym('v',3)
    phi = SX.sym('phi')
    theta = SX.sym('theta')
    psi = SX.sym('psi')

    gravity = SX.zeros(3)
    gravity[2] = g

    x = vertcat(
        position,
        phi,
        theta,
        psi,
        velocity
    )

    p = SX.sym('p')
    q = SX.sym('q')
    r = SX.sym('r')
    F = SX.sym('F')

    u = vertcat(
        p,
        q,
        r,
        F
    )

    r1 = SX(3,3)
    r1[0,0] = 1
    r1[1,1] = cos(phi)
    r1[1,2] = -sin(phi)
    r1[2,1] = sin(phi)
    r1[2,2] = cos(phi)

    r2 = SX(3,3)
    r2[0,0] = cos(theta)
    r2[0,2] = sin(theta)
    r2[1,1] = 1
    r2[2,0] = -sin(theta)
    r2[2,2] = cos(theta)

    r3 = SX(3,3)
    r3[0,0] = cos(psi)
    r3[0,1] = -sin(psi)
    r3[1,0] = sin(psi)
    r3[1,1] = cos(psi)
    r3[2,2] = 1

    tau = SX(3,3)
    tau[:,0] = (inv(r2))[:,0]
    tau[:,1] = (inv(r2))[:,1]
    tau[:,2] = (inv(r2 @ r1))[:,2]
    # tau[:,0] = (inv(r1))[:,0]
    # tau[:,1] = (inv(r1))[:,1]
    # tau[:,2] = (inv(r1 @ r2))[:,2]

    # tau[0,0] = cos(theta)
    # tau[0,2] = -sin(theta)
    # tau[1,1] = 1
    # tau[1,2] = sin(phi) * cos(theta)
    # tau[2,0] = sin(theta)
    # tau[2,2] = cos(phi) * cos(theta)

    tc = 0.59375 # mapping control to throttle
    # f_const = 100
    f_const = 20
    # tc = 0.75
    #
    #  scaling_control_mat = np.array([1, 1, 1, (g/tc)])

    # f_expl = vertcat(
    #     velocity,
    #     inv(tau) @ u[0:3], 
    #     - ((r3 @ r1 @ r2)[:,2]) * (u[3] * ((g)/tc)) + gravity
    # )

    f_expl = vertcat(
        velocity,
        inv(tau) @ u[0:3], 
        - ((r3 @ r1 @ r2)[:,2]) * (u[3] * ((g + f_const)/tc) - f_const) + gravity
    )
    
    # f_expl = vertcat(
    #     velocity,
    #     inv(tau) @ u[0:3], 
    #     - ((r3 @ r1 @ r2)[:,2]) * ((g - f_const)*(1 - tc) / (1 - u[3]) + f_const)  + gravity
    # )

    # xdot
    xdot = SX.sym('xdot',9)

    #parameters
    p = []

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model