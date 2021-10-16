import numpy as np
import sys

class ForwardBackwardSweep:

    def __init__(self, state_eqs, costate_eqs, optimality_condition, u_0, x_init, lambda_final,
            args_st = () , args_cos = (), args_cont = (), t_0=0.0, t_f=1.0, 
            Ntotal = 500, iterationmax = 200, error_history = False):
        
        self.x_zero = x_init                                # initial conditions
        self.number_var = len(self.x_zero)                  # number of x variables
        self.iterationmax = iterationmax                    # maximum number of iteration for computing the solution.
        self.error_history = error_history                  # to return or not the errors of the computation.
        self.lambda_final = lambda_final                    # final conditions of the constraints
        self.state_eqs = state_eqs                          # function for the state equations
        self.costate_eqs = costate_eqs                      # function for the costate equations
        self.optimality_condition = optimality_condition    # function for the optimality condition
        self.u_0 = u_0                                      # first guess in the control path u
        self.t_0 = t_0                                      # initial time
        self.t_f = t_f                                      # final time
        self.Ntotal = Ntotal                                # total number of points in the mesh
        t = np.linspace(self.t_0, self.t_f, self.Ntotal)    # time mesh
        self.h = t[1] - t[0]                                # time step size
        self.args_st = args_st                              # aditional arguments of the state equations
        self.args_cos = args_cos                            # aditional arguments of the costate equations
        self.args_cont = args_cont                          # aditional arguments of the optimality condition

    def runge_kutta_forward(self, u):
        sol = np.zeros((self.number_var, self.Ntotal))      # initialize solution matrix. Size number of systems x time mesh size
        sol[:,0] = self.x_zero                              # initial conditions
        
        # check that the initial guess has the same dimension as x_zero otherwise 
        # print an error message
    
        for j in np.arange(self.Ntotal-1):                  # forward loop

            x_j = sol[:,j] 
            u_j = u[:,j]
            u_jp1 = u[:,j + 1]
            u_mj = 0.5 * (u_j + u_jp1)                      # interpolation to calculate the value of u at j+0.5h
            
            k_1 = self.state_eqs(x_j, u_j, *self.args_st)
            k_2 = self.state_eqs(x_j + 0.5 * self.h * k_1, u_mj, *self.args_st)
            k_3 = self.state_eqs(x_j + 0.5 * self.h * k_2, u_mj, *self.args_st)
            k_4 = self.state_eqs(x_j + self.h * k_3, u_jp1, *self.args_st)

            sol[:, j + 1] = x_j + (self.h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

        return sol

    def runge_kutta_backward(self, x, u):
        sol = np.zeros((self.number_var, self.Ntotal))      # initialize solution matrix
        sol[:,-1] = self.lambda_final                       # final conditions
        
        #check that the initial guess has the same dimension as x_zero otherwise 
        #print an error message
    
        for j in np.arange(self.Ntotal-1, 0, -1):            # backward loop

            lambda_j = sol[:,j]
            x_j = x[:,j]
            x_jm1 = x[:,j - 1]
            x_mj = 0.5 * (x_j + x_jm1)                       # interpolation to calculate the value of x at j-0.5h
            u_j = u[:,j]
            u_jm1 = u[:,j - 1] 
            u_mj = 0.5 * (u_j + u_jm1)                       # interpolation to calculate the value of u at j-0.5h
            
            k_1 = self.costate_eqs(lambda_j, x_j, u_j, *self.args_cos)
            k_2 = self.costate_eqs(lambda_j - 0.5 * self.h * k_1, x_mj, u_mj, *self.args_cos)
            k_3 = self.costate_eqs(lambda_j - 0.5 * self.h * k_2, x_mj, u_mj, *self.args_cos)
            k_4 = self.costate_eqs(lambda_j - self.h * k_3, x_jm1, u_jm1, *self.args_cos)
            
            sol[:, j - 1] = lambda_j - (self.h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        return sol
    
    def solve(self):
        i = 0
        flag = True
        # dictionary to store the errors
        # the range(3) accounts for the state, costate and control vectors 
        errors = { str(i) : [[] for j in range(3) ] for i in range(self.number_var)} 
        test = []
        x = np.zeros((self.number_var, self.Ntotal))         # initialize solution matrix for x
        x[:,0] = self.x_zero                                 # initial conditions
        lmbda = np.zeros((self.number_var, self.Ntotal))     # initialize solution matrix for lambda
        lmbda[:,-1] = self.lambda_final                      # final conditions
        
        if self.u_0.shape[0] != self.number_var: sys.exit("Error: The row length of the control matrix 'u' is not correct")
        elif self.u_0.shape[1] != self.Ntotal: sys.exit("Error: The column length of the control matrix 'u' is not correct")
        else:     
            u = self.u_0                                     # initialize control path u
            for _ in range(self.iterationmax):  

                u_old = u 
                x_old = x 
                lambda_old = lmbda
                x = self.runge_kutta_forward(u)              # compute forward solution
                lmbda = self.runge_kutta_backward(x,u)       # compute backward solution with the new x
                u_1 = self.optimality_condition(x, u, lmbda, *self.args_cont) # compute the new controller
                u = 0.5 * (u_1 + u_old)                      # update the controller. Convex combination between the previous
                                                             # and current controller. This helps to the speed of convergence.

                # computing the errors
                for j in range(self.number_var):
                    error_in_u = np.linalg.norm(u_old[j] - u[j], 1) * (np.linalg.norm(u[j], 1) ** (-1))
                    error_in_x = np.linalg.norm(x_old[j] - x[j], 1) * (np.linalg.norm(x[j], 1) ** (-1))
                    error_in_lambda = np.linalg.norm(lambda_old[j] - lmbda[j], 1) * (np.linalg.norm(lmbda[j], 1) ** (-1))
                    
                    # fill the dictionary. j that indicates the key number.
                    errors[str(j)][0].append(error_in_x)
                    errors[str(j)][1].append(error_in_lambda)
                    errors[str(j)][2].append(error_in_u)
                    
                i += 1
            
                print("Iteration {} completed.".format(i))
            
            return (x, lmbda, u, errors) if self.error_history == True else (x, lmbda, u)