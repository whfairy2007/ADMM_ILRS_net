# network model
import numpy as np
class ADMM_IRLS_Net(object):
    # def __init__(self, X, y, dtype=np.float32):
    def __init__(self, dtype=np.float32):
        """
        X
        y
        """
        self.y = []
        self.X = []
        self.Phi_bar = []
        self.N = []
        self.M = []
        self.N_layer = 50  # network size
        self.N_node_type = 19  # #node type
        self.dtype = dtype
        # network structure
        self.net_struct = {'Layer1': 'node_h', 'Layer2': 'node_m', 'Layer3': 'node_l', 'Layer4': 'node_t',
                           'Layer5': 'node_i',
                           'Layer6': 'node_m', 'Layer7': 'node_k', 'Layer8': 'node_q', 'Layer9': 'node_i',
                           'Layer10': 'node_m',
                           'Layer11': 'node_k', 'Layer12': 'node_r', 'Layer13': 'node_j', 'Layer14': 'node_o',
                           'Layer15': 'node_c',
                           'Layer16': 'node_b', 'Layer17': 'node_a', 'Layer18': 'node_d', 'Layer19': 'node_g',
                           'Layer20': 'node_m',
                           'Layer21': 'node_l', 'Layer22': 'node_t', 'Layer23': 'node_e', 'Layer24': 'node_m',
                           'Layer25': 'node_k',
                           'Layer26': 'node_q', 'Layer27': 'node_e', 'Layer28': 'node_m', 'Layer29': 'node_k',
                           'Layer30': 'node_r',
                           'Layer31': 'node_f', 'Layer32': 'node_n', 'Layer33': 'node_c', 'Layer34': 'node_b',
                           'Layer35': 'node_a',
                           'Layer36': 'node_d', 'Layer37': 'node_g', 'Layer38': 'node_m', 'Layer39': 'node_l',
                           'Layer40': 'node_t',
                           'Layer41': 'node_e', 'Layer42': 'node_m', 'Layer43': 'node_k', 'Layer44': 'node_q',
                           'Layer45': 'node_e',
                           'Layer46': 'node_m', 'Layer47': 'node_k', 'Layer48': 'node_r', 'Layer49': 'node_f',
                           'Layer50': 'node_p'}

        # values of k and i in each layer. These are used to find correct variables
        self.node_kvalue = np.concatenate((np.ones(18), 2 * np.ones(18), 3 * np.ones(14)), axis=0)

        self.node_ivalue = np.concatenate((np.ones(4), 2 * np.ones(4), 3 * np.ones(4), 4 * np.ones(6), np.ones(4),
                                           2 * np.ones(4), 3 * np.ones(4), 4 * np.ones(6),
                                           np.ones(4), 2 * np.ones(4), 3 * np.ones(4), 4 * np.ones(2)), axis=0)

        # state of variables
        self.state_var = {'Omega_1_1': [], 'C_1_1': [], 'z_1_1': [], 'u_1_1': [], 'Omega_1_2': [], 'C_1_2': [],
                          'z_1_2': [],
                          'u_1_2': [], 'Omega_1_3': [], 'C_1_3': [], 'z_1_3': [], 'u_1_3': [], 'Omega_1_4': [],
                          'Belta_1': [],
                          'v_1': [], 'Sigma_1': [], 'A_1': [], 'b_1': [], 'Omega_2_1': [], 'C_2_1': [], 'z_2_1': [],
                          'u_2_1': [],
                          'Omega_2_2': [], 'C_2_2': [], 'z_2_2': [], 'u_2_2': [], 'Omega_2_3': [], 'C_2_3': [],
                          'z_2_3': [],
                          'u_2_3': [], 'Omega_2_4': [], 'Belta_2': [], 'v_2': [], 'Sigma_2': [], 'A_2': [], 'b_2': [],
                          'Omega_3_1': [],
                          'C_3_1': [], 'z_3_1': [], 'u_3_1': [], 'Omega_3_2': [], 'C_3_2': [], 'z_3_2': [], 'u_3_2': [],
                          'Omega_3_3': [], 'C_3_3': [], 'z_3_3': [], 'u_3_3': [], 'Omega_3_4': [], 'Belta_3': []}
        # grandients of variables and states
        self.gradient = {}

        # initialize parameters : F >0, rho>0, 0<t<1
        self.params = {'F_1_1': [], 'rho_1_1': [], 'F_1_2': [], 'rho_1_2': [], 'F_1_3': [],
                       'rho_1_3': [], 'F_1_4': [], 'rho_1_4': [], 't_1': [],
                       'F_2_1': [], 'rho_2_1': [], 'F_2_2': [], 'rho_2_2': [], 'F_2_3': [],
                       'rho_2_3': [], 'F_2_4': [], 'rho_2_4': [], 't_2': [],
                       'F_3_1': [], 'rho_3_1': [], 'F_3_2': [], 'rho_3_2': [], 'F_3_3': [],
                       'rho_3_3': [], 'F_3_4': [], 'rho_3_4': [], 't_3': []}

        self.grads = {'dE_dF_1_1': [], 'dE_drho_1_1': [], 'dE_dF_1_2': [], 'dE_drho_1_2': [], 'dE_dF_1_3': [],
                      'dE_drho_1_3': [], 'dE_dF_1_4': [], 'dE_drho_1_4': [], 'dE_dt_1': [],
                      'dE_dF_2_1': [], 'dE_drho_2_1': [], 'dE_dF_2_2': [], 'dE_drho_2_2': [], 'dE_dF_2_3': [],
                      'dE_drho_2_3': [], 'dE_dF_2_4': [], 'dE_drho_2_4': [], 't_2': [],
                      'dE_dF_3_1': [], 'dE_drho_3_1': [], 'dE_dF_3_2': [], 'dE_drho_3_2': [], 'dE_dF_3_3': [],
                      'dE_drho_3_3': [], 'dE_dF_3_4': [], 'dE_drho_3_4': [], 'dE_dt_3': []}

    def loss(self, X, y=None):
        """
        Compute loss and gradient
        """
        self.X = X.astype(self.dtype)

        self.N, self.M = np.shape(X)
        M = self.M
        if np.size(self.params['F_1_1']) == 0:
            rho0 = 1
            t0 = 1
            self.params = {'F_1_1': 10 * np.identity(M), 'rho_1_1': rho0, 'F_1_2': 10 * np.identity(M), 'rho_1_2': rho0,
                           'F_1_3': 10 * np.identity(M),
                           'rho_1_3': rho0, 'F_1_4': 10 * np.identity(M), 'rho_1_4': rho0, 't_1': t0,
                           'F_2_1': 10 * np.identity(M), 'rho_2_1': rho0, 'F_2_2': 10 * np.identity(M), 'rho_2_2': rho0,
                           'F_2_3': 10 * np.identity(M),
                           'rho_2_3': rho0, 'F_2_4': 10 * np.identity(M), 'rho_2_4': rho0, 't_2': t0,
                           'F_3_1': 10 * np.identity(M), 'rho_3_1': rho0, 'F_3_2': 10 * np.identity(M), 'rho_3_2': rho0,
                           'F_3_3': 10 * np.identity(M),
                           'rho_3_3': rho0, 'F_3_4': 10 * np.identity(M), 'rho_3_4': rho0, 't_3': t0}




            # mode = 'test' if y is None else 'train'
        if y is None:
            self.scores = np.dot(X, self.state_var['Belta_3'])
            return self.scores
        """
        Implement the forward pass
        """
        self.Phi_bar = np.dot(np.diagflat(y), X)
        self.y = y
        for l in range(50):
            # for l in range(self.N_layer):
            Layer_name = 'Layer' + str(l + 1)
            node_name = self.net_struct[Layer_name]

            node_func = eval('self.' + node_name + '_forward')

            node_func(l)

        self.E, self.scores = self.network_loss()  # loss

        ##
        loss, grads = 0.0, {}

        """
        Implement the backpropogation
        """
        # layer_index = list(range(49,-1,-1))
        layer_index = list(range(49, -1, -1))
        for l in layer_index:
            Layer_name = 'Layer' + str(l + 1)
            node_name = self.net_struct[Layer_name]

            node_func = eval('self.' + node_name + '_backward')

            node_func(l)
            # gradient is save in self.gradient
        ######################################################
        loss = self.E
        grads = self.grads

        return loss, grads

    def network_loss(self):
        Belta_3 = self.state_var['Belta_3']
        y = self.y
        X = self.X
        E = 0
        for n, y_n in enumerate(y):
            E += np.log(1 + np.exp(-y_n * np.dot(X[n,], Belta_3)))
        scores = np.dot(X, Belta_3)
        return E, scores

    def node_p_backward(self, l):
        """
        Compute the backward pass for node p. Layer 50

        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        y = self.y
        X = self.X
        M = self.M
        N = self.N
        Belta_k = self.state_var['Belta_' + str(k)]
        Belta_k_1 = self.state_var['Belta_' + str(k - 1)]
        Omega_k_i = self.state_var['Omega_' + str(k) + '_' + str(i)]
        dE_dBelta_k = np.zeros((1, M))  # row vector
        for n, y_n in enumerate(y):
            dBelta_k1 = y_n * np.dot(X[n,], Belta_k)
            dBelta_k2 = y_n * np.exp(-dBelta_k1) / (1 + np.exp(-dBelta_k1))
            dBelta_k3 = dBelta_k2 * X[n,]
            dE_dBelta_k -= dBelta_k3
        self.gradient['dE_dBelta_' + str(k)] = dE_dBelta_k
        self.grads['dE_dBelta_' + str(k)] = dE_dBelta_k
        t_k = self.params['t_' + str(k)]
        dBelta_k_dOmega_k_i = np.multiply(t_k, np.eye(M))
        self.gradient['dBelta_' + str(k) + '_dOmega_' + str(k) + '_' + str(i)] = dBelta_k_dOmega_k_i
        dBelta_k_dBelta_k_1 = np.multiply((1 - t_k), np.eye(M))
        self.gradient['dBelta_' + str(k) + '_dBelta_' + str(k - 1)] = dBelta_k_dBelta_k_1

        # output/parameter
        dBelta_k_dt_k = -Belta_k_1 + Omega_k_i
        self.gradient['dBelta_' + str(k) + '_dt_' + str(k)] = dBelta_k_dt_k
        # loss/parameter
        dE_dt_k = np.dot(dE_dBelta_k, dBelta_k_dt_k)
        self.gradient['dE_dt_' + str(k)] = dE_dt_k

        self.grads['dE_dt_' + str(k)] = dE_dt_k

    def node_p_forward(self, l):
        """
        Compute the forward pass for node p. Layer 50
        forward pass is same as node n
        """
        self.node_n_forward(l)

    def node_n_backward(self, l):
        """
        Compute the backward pass for node n. Layer 32
        """
        self.node_n_o_backward(l)

    def node_n_o_backward(self, l):
        """
        Compute the backward pass for node n and o
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        M = self.M
        N = self.N
        Omega_k_i = self.state_var['Omega_' + str(k) + '_' + str(i)]
        Layer_name = 'Layer' + str(l + 1)
        node_name = self.net_struct[Layer_name]
        if node_name == 'node_o':
            Belta_k_1 = np.zeros((M, 1))
        else:
            Belta_k_1 = self.state_var['Belta_' + str(k - 1)]

        dE_dSigma_k = np.diag(self.gradient['dE_dSigma_' + str(k)])  # (N,)
        dSigma_k_dBelta_k = self.gradient['dSigma_' + str(k) + '_dBelta_' + str(k)]  # (N,M)
        dE_dv_k = self.gradient['dE_dv_' + str(k)]
        dv_k_dBelta_k = self.gradient['dv_' + str(k) + '_dBelta_' + str(k)]
        dE_dBelta_k_1 = self.gradient['dE_dBelta_' + str(k + 1)]
        dBelta_k_1_dBelta_k = self.gradient['dBelta_' + str(k + 1) + '_dBelta_' + str(k)]
        # loss / output
        dE_dBelta_k = np.dot(dE_dSigma_k, dSigma_k_dBelta_k) + np.dot(dE_dv_k, dv_k_dBelta_k) + np.dot(dE_dBelta_k_1,
                                                                                                       dBelta_k_1_dBelta_k)
        self.gradient['dE_dBelta_' + str(k)] = dE_dBelta_k
        self.grads['dE_dBelta_' + str(k)] = dE_dBelta_k

        # output/parameter
        dBelta_k_dt_k = -Belta_k_1 + Omega_k_i
        self.gradient['dBelta_' + str(k) + '_dt_' + str(k)] = dBelta_k_dt_k
        # loss/parameter
        dE_dt_k = np.dot(dE_dBelta_k, dBelta_k_dt_k)
        self.gradient['dE_dt_' + str(k)] = dE_dt_k
        self.grads['dE_dt_' + str(k)] = dE_dt_k

        # output/input
        t_k = self.params['t_' + str(k)]
        dBelta_k_dOmega_k_i = np.multiply(t_k, np.eye(M))
        self.gradient['dBelta_' + str(k) + '_dOmega_' + str(k) + '_' + str(i)] = dBelta_k_dOmega_k_i
        if node_name == 'node_n':
            dBelta_k_dBelta_k_1 = np.multiply((1 - t_k), np.eye(M))
            self.gradient['dBelta_' + str(k) + '_dBelta_' + str(k - 1)] = dBelta_k_dBelta_k_1

            #####################################################

    def node_n_forward(self, l):
        """
        Compute the forward pass for node n. Layer 32
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        Omega_k_i = self.state_var['Omega_' + str(k) + '_' + str(i)]
        Belta_k_1 = self.state_var['Belta_' + str(k - 1)]
        t_k = self.params['t_' + str(k)]
        Belta_k = np.multiply((1 - t_k), Belta_k_1) + np.multiply(t_k, Omega_k_i)
        self.state_var['Belta_' + str(k)] = Belta_k
        # print(Belta_k_1.shape)

    def node_f_backward(self, l):
        """
        Compute the forward pass for node f. Layer 49
        """
        self.node_e_f_g_i_j_h_backward(l)

    def node_f_forward(self, l):
        """
        Compute the forward pass for node f. Layer 31,49
        forward pass is same as node e
        """
        self.node_e_forward(l)

    def node_e_backward(self, l):
        """
        Compute the backward pass for node e. Layer 23,27,41,45
        """
        self.node_e_f_g_i_j_h_backward(l)

    def node_e_forward(self, l):
        """
        Compute the forward pass for node e. Layer 23,27,41,45
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        A_k_1 = self.state_var['A_' + str(k - 1)]
        z_k_i_1 = self.state_var['z_' + str(k) + '_' + str(i - 1)]
        u_k_i_1 = self.state_var['u_' + str(k) + '_' + str(i - 1)]
        b_k_1 = self.state_var['b_' + str(k - 1)]
        F_k_i = self.params['F_' + str(k) + '_' + str(i)]
        rho_k_i = self.params['rho_' + str(k) + '_' + str(i)]

        temp1 = np.multiply(rho_k_i, np.dot(np.transpose(F_k_i), F_k_i))
        Omega_k_i_1 = np.linalg.inv(np.dot(np.transpose(A_k_1), A_k_1) + temp1)
        Omega_k_i_2 = np.dot(np.transpose(A_k_1), b_k_1) + np.multiply(rho_k_i,
                                                                       np.dot(np.transpose(F_k_i), (z_k_i_1 - u_k_i_1)))
        Omega_k_i = np.dot(Omega_k_i_1, Omega_k_i_2)
        self.state_var['Omega_' + str(k) + '_' + str(i)] = Omega_k_i

    def node_g_backward(self, l):
        """
        Compute the backward pass for node g. Layer 19,37
        """
        self.node_e_f_g_i_j_h_backward(l)

    def node_g_forward(self, l):
        """
        Compute the forward pass for node g. Layer 19,37
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        rho_k_i = self.params['rho_' + str(k) + '_' + str(i)]
        F_k_i = self.params['F_' + str(k) + '_' + str(i)]
        b_k_1 = self.state_var['b_' + str(k - 1)]
        A_k_1 = self.state_var['A_' + str(k - 1)]

        temp1 = np.multiply(rho_k_i, np.dot(np.transpose(F_k_i), F_k_i))
        Omega_k_i_1 = np.linalg.inv(np.dot(np.transpose(A_k_1), A_k_1) + temp1)
        Omega_k_i = np.dot(np.dot(Omega_k_i_1, np.transpose(A_k_1)), b_k_1)
        self.state_var['Omega_' + str(k) + '_' + str(i)] = Omega_k_i

    ###############################################################################################
    def node_d_backward(self, l):
        """
        Compute the backward pass for node d. Layer 18,36
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        M = self.M
        N = self.N
        # loss/output
        dE_db_k = 0
        for idx in range(4):
            i2 = idx + 1
            dE_dOmega_k_i = self.gradient['dE_dOmega_' + str(k + 1) + '_' + str(i2)]
            dOmega_k_1_i_db_k = self.gradient['dOmega_' + str(k + 1) + '_' + str(i2) + '_db_' + str(k)]
            dE_db_k += np.dot(dE_dOmega_k_i, dOmega_k_1_i_db_k)
        self.gradient['dE_db_' + str(k)] = dE_db_k
        self.grads['dE_db_' + str(k)] = dE_db_k

        # output/input
        Sigma_k = self.state_var['Sigma_' + str(k)]
        v_k = self.state_var['v_' + str(k)]

        db_k_dSigma_k = np.zeros((N, N))
        for n in range(N):
            # I_bar = np.zeros((N,N))
            # I_bar[n,n] = 1
            # db_k_dSigma_k[:,[n]] = 0.5*np.dot(np.dot(Sigma_k**(-0.5),I_bar),v_k)
            I_bar = np.zeros((N, N))
            I_bar[n, n] = Sigma_k[n, n] ** (-0.5)
            db_k_dSigma_k[:, [n]] = 0.5 * np.dot(I_bar, v_k)

        self.gradient['db_' + str(k) + '_dSigma_' + str(k)] = db_k_dSigma_k
        self.gradient['db_' + str(k) + '_dv_' + str(k)] = 0.5 * Sigma_k ** (0.5)

    ###############################################################################################

    def node_d_forward(self, l):
        """
        Compute the forward pass for node d. Layer 18,36
        """
        k = int(self.node_kvalue[l])
        Sigma_k = self.state_var['Sigma_' + str(k)]
        v_k = self.state_var['v_' + str(k)]
        b_k = np.dot(Sigma_k ** 0.5, v_k)
        self.state_var['b_' + str(k)] = b_k

    def node_a_backward(self, l):
        """
        Compute the backward pass for node a. Layer 17,35
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        N = self.N
        M = self.M
        Phi_bar = self.Phi_bar
        dOmega_k_i_1_dA_k = self.gradient['dOmega_' + str(k + 1) + '_' + str(i) + '_dA_' + str(k)]
        # loss/output
        dE_dA_k = np.zeros((M, N))
        for nx in range(N):
            for ny in range(M):
                dE_dA_k_nx_ny = 0
                for idx in range(4):
                    i2 = idx + 1
                    dE_dOmega_k_i = self.gradient['dE_dOmega_' + str(k + 1) + '_' + str(i2)]
                    dE_dA_k_nx_ny += np.dot(dE_dOmega_k_i, dOmega_k_i_1_dA_k[:, [ny], [nx]])
                dE_dA_k[ny, nx] = dE_dA_k_nx_ny
        self.gradient['dE_dA_' + str(k)] = dE_dA_k
        self.grads['dE_dA_' + str(k)] = dE_dA_k
        ##############################################
        # dA_k_dSigma_k = np.zeros((N,N,M))
        # Sigma_k = self.state_var['Sigma_'+str(k)]
        # for n in range(N):
        #   I_bar = np.zeros((N,N))
        #    I_bar[n,n] = Sigma_k[n,n]**(-0.5)
        #   dA_k_dSigma_k[[n],:,:]  = 0.5*np.dot(I_bar,Phi_bar)
        ############################################### dimension reduced version
        dA_k_dSigma_k = np.zeros((N, M))
        Sigma_k = self.state_var['Sigma_' + str(k)]
        for n in range(N):
            dA_k_dSigma_k[[n], :] = 0.5 * Sigma_k[n, n] ** (-0.5) * Phi_bar[n, :]
        self.gradient['dA_' + str(k) + '_dSigma_' + str(k)] = dA_k_dSigma_k

    def node_a_forward(self, l):
        """
        Compute the forward pass for node a. Layer 17,35
        """
        k = int(self.node_kvalue[l])
        Phi_bar = self.Phi_bar
        Sigma_k = self.state_var['Sigma_' + str(k)]
        A_k = np.dot(Sigma_k ** 0.5, Phi_bar)
        self.state_var['A_' + str(k)] = A_k

    def node_b_backward(self, l):
        """
        Compute the backward pass for node b. Layer 16,34
        """
        k = int(self.node_kvalue[l])
        Belta_k = self.state_var['Belta_' + str(k)]
        y = self.y
        X = self.X
        M = self.M
        N = self.N
        dE_dA_k = self.gradient['dE_dA_' + str(k)]
        dA_k_dSigma_k = self.gradient['dA_' + str(k) + '_dSigma_' + str(k)]
        db_k_dSigma_k = self.gradient['db_' + str(k) + '_dSigma_' + str(k)]
        dE_db_k = self.gradient['dE_db_' + str(k)]
        dE_dSigma_k = np.zeros((N, N))
        for n in range(N):
            dE_dSigma_k[n, n] = np.dot(dA_k_dSigma_k[[n], :], dE_dA_k[:, [n]]) + np.dot(dE_db_k, db_k_dSigma_k[:, [n]])
        self.gradient['dE_dSigma_' + str(k)] = dE_dSigma_k
        self.grads['dE_dSigma_' + str(k)] = dE_dSigma_k

        dSigma_k_dBelta_k = np.zeros((N, M))
        for n in range(N):
            n1 = np.exp(y[n] * np.dot(X[n, :], Belta_k))
            n2 = np.exp(-y[n] * np.dot(X[n, :], Belta_k))
            val_n = y[n] * (n1 / (1 + n1) ** 3 - n2 / (1 + n2) ** 3)
            dSigma_k_dBelta_k[[n], :] = val_n * X[n, :]
        self.gradient['dSigma_' + str(k) + '_dBelta_' + str(k)] = dSigma_k_dBelta_k

    def node_b_forward(self, l):
        """
        Compute the forward pass for node b. Layer 16,34
        """
        k = int(self.node_kvalue[l])
        Belta_k = self.state_var['Belta_' + str(k)]
        y = self.y
        X = self.X
        M = self.M
        N = self.N
        Sigma_k = np.zeros((N, N))
        for n, y_n in enumerate(y):
            Sigma_1 = y_n * np.dot(X[n,], Belta_k)
            Sigma_k[n, n] = np.exp(-Sigma_1) / (1 + np.exp(-Sigma_1)) ** 2
        self.state_var['Sigma_' + str(k)] = Sigma_k

    def node_c_backward(self, l):
        """
        Compute the backward pass for node c. Layer 15,33
        """
        k = int(self.node_kvalue[l])
        y = self.y
        X = self.X
        M = self.M
        N = self.N
        Belta_k = self.state_var['Belta_' + str(k)]
        dE_db_k = self.gradient['dE_db_' + str(k)]
        db_k_dv_k = self.gradient['db_' + str(k) + '_dv_' + str(k)]
        dE_dv_k = np.dot(dE_db_k, db_k_dv_k)
        self.gradient['dE_dv_' + str(k)] = dE_dv_k
        self.grads['dE_dv_' + str(k)] = dE_dv_k

        dv_k_dBelta_k = np.zeros((N, M))
        for n in range(N):
            for m in range(M):
                dv_k_dBelta_k[n, m] = y[n] * X[n, m] * (1 - np.exp(-y[n] * np.dot(X[n, :], Belta_k)))
        self.gradient['dv_' + str(k) + '_dBelta_' + str(k)] = dv_k_dBelta_k


        #######################################################

    def node_c_forward(self, l):
        """
        Compute the forward pass for node c. Layer 15,33
        """
        k = int(self.node_kvalue[l])
        Belta_k = self.state_var['Belta_' + str(k)]
        y = self.y
        X = self.X
        v_k = np.zeros(y.shape)
        for n, y_n in enumerate(y):
            v_1 = y_n * np.dot(X[n, :], Belta_k)
            v_k[n] = v_1 + 1 + np.exp(-v_1)
        self.state_var['v_' + str(k)] = v_k

    def node_o_backward(self, l):
        """
        Compute the backward pass for node o. Layer 14
        """
        self.node_n_o_backward(l)

    def node_o_forward(self, l):
        """
        Compute the forward pass for node o. Layer 14
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        Omega_k_i = self.state_var['Omega_' + str(k) + '_' + str(i)]
        t_k = self.params['t_' + str(k)]
        Belta_k = np.multiply(t_k, Omega_k_i)
        self.state_var['Belta_' + str(k)] = Belta_k
        # print(Belta_k.shape)

    def node_j_backward(self, l):
        """
        Compute the backward pass for node j. Layer 13
        """
        self.node_e_f_g_i_j_h_backward(l)

    def node_j_forward(self, l):
        """
        Compute the forward pass for node j. Layer 13
        forward pass is same as node i
        """
        self.node_i_forward(l)

    def node_r_q_t_backward(self, l):
        """
        Compute the backward pass for node r, q,t: common part
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        M = self.M
        N = self.N
        Layer_name = 'Layer' + str(l + 1)
        node_name = self.net_struct[Layer_name]
        dOmega_k_i_1_du_k_i = self.gradient['dOmega_' + str(k) + '_' + str(i + 1) + '_du_' + str(k) + '_' + str(i)]
        dE_dOmega_k_i_1 = self.gradient['dE_dOmega_' + str(k) + '_' + str(i + 1)]
        if node_name == 'node_r':
            dE_du_k_i = np.dot(dE_dOmega_k_i_1, dOmega_k_i_1_du_k_i)
        else:  # 'node_q','node_t'
            dE_du_k_i_1 = self.gradient['dE_du_' + str(k) + '_' + str(i + 1)]
            du_k_i_1_du_k_i = self.gradient['du_' + str(k) + '_' + str(i + 1) + '_du_' + str(k) + '_' + str(i)]
            dE_dz_k_i_1 = self.gradient['dE_dz_' + str(k) + '_' + str(i + 1)]
            dz_k_i_1_du_k_i = self.gradient[
                'dz_' + str(k) + '_' + str(i + 1) + '_du_' + str(k) + '_' + str(i)]  #############
            dE_du_k_i = np.dot(dE_du_k_i_1, du_k_i_1_du_k_i) + np.dot(dE_dz_k_i_1, dz_k_i_1_du_k_i) + np.dot(
                dE_dOmega_k_i_1, dOmega_k_i_1_du_k_i)
        self.gradient['dE_du_' + str(k) + '_' + str(i)] = dE_du_k_i
        self.grads['dE_du_' + str(k) + '_' + str(i)] = dE_du_k_i
        # output/input
        if node_name != 'node_t':
            self.gradient['du_' + str(k) + '_' + str(i) + '_du_' + str(k) + '_' + str(i - 1)] = np.eye(
                M)  # du_k_i_du_k_i_1
        self.gradient['du_' + str(k) + '_' + str(i) + '_dC_' + str(k) + '_' + str(i)] = np.eye(M)  # du_k_i_dC_k_i
        self.gradient['du_' + str(k) + '_' + str(i) + '_dz_' + str(k) + '_' + str(i)] = - np.eye(
            M)  # du_k_i_dz_k_i

    def node_r_backward(self, l):
        """
        Compute the backward pass for node r. Layer 12,30,48
        """
        self.node_r_q_t_backward(l)

    def node_r_forward(self, l):
        """
        Compute the forward pass for node r. Layer 12,30,48
        forward pass is same as node q
        """
        self.node_q_forward(l)

    def node_q_backward(self, l):
        """
        Compute the backward pass for node q. Layer 8,26,44
        """
        self.node_r_q_t_backward(l)

    def node_q_forward(self, l):
        """
        Compute the forward pass for node q. Layer 8,26,44
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        u_k_i_1 = self.state_var['u_' + str(k) + '_' + str(i - 1)]
        C_k_i = self.state_var['C_' + str(k) + '_' + str(i)]
        z_k_i = self.state_var['z_' + str(k) + '_' + str(i)]
        u_k_i = u_k_i_1 + C_k_i - z_k_i
        self.state_var['u_' + str(k) + '_' + str(i)] = u_k_i

    def node_k_l_backward(self, l):
        """
        Compute the forward pass for node k and l: common part
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        C_k_i = self.state_var['C_' + str(k) + '_' + str(i)]
        M = self.M
        N = self.N
        Layer_name = 'Layer' + str(l + 1)
        node_name = self.net_struct[Layer_name]
        if node_name == 'node_k':
            u_k_i_1 = self.state_var['u_' + str(k) + '_' + str(i - 1)]
        else:
            u_k_i_1 = np.zeros((M, 1))

        rho_k_i = self.params['rho_' + str(k) + '_' + str(i)]

        dE_du_k_i = self.gradient['dE_du_' + str(k) + '_' + str(i)]
        du_k_i_dz_k_i = self.gradient['du_' + str(k) + '_' + str(i) + '_dz_' + str(k) + '_' + str(i)]
        dE_dOmega_k_i_1 = self.gradient['dE_dOmega_' + str(k) + '_' + str(i + 1)]
        dOmega_k_i_1_dz_k_i = self.gradient['dOmega_' + str(k) + '_' + str(i + 1) + '_dz_' + str(k) + '_' + str(i)]
        # loss/output
        dE_dz_k_i = np.dot(dE_du_k_i, du_k_i_dz_k_i) + np.dot(dE_dOmega_k_i_1, dOmega_k_i_1_dz_k_i)
        self.gradient['dE_dz_' + str(k) + '_' + str(i)] = dE_dz_k_i
        self.grads['dE_dz_' + str(k) + '_' + str(i)] = dE_dz_k_i

        # output/parameter
        dz_k_i_drho_k_i = np.zeros((M, 1))
        dz_k_i_dC_k_i = np.zeros((M, M))  # diagonal
        dz_k_i_du_k_i_1 = np.zeros((M, M))  # diagonal
        for m in range(M):
            dz_k_i_drho_k_i[m] = 0
            dz_k_i_dC_k_i[m, m] = 0
            dz_k_i_du_k_i_1[m, m] = 0
            cmp_m = C_k_i[m] + u_k_i_1[m]
            if cmp_m > 1 / rho_k_i:
                dz_k_i_drho_k_i[m] = rho_k_i ** (-2)
                dz_k_i_dC_k_i[m, m] = 1
                dz_k_i_du_k_i_1[m, m] = 1
            elif cmp_m < -1 / rho_k_i:
                dz_k_i_drho_k_i[m] = -rho_k_i ** (-2)
                dz_k_i_dC_k_i[m, m] = 1
                dz_k_i_du_k_i_1[m, m] = 1
        self.gradient['dz_' + str(k) + '_' + str(i) + '_drho_' + str(k) + '_' + str(i)] = dz_k_i_drho_k_i
        # loss/parameter
        dE_drho_k_i = np.dot(dE_dz_k_i, dz_k_i_drho_k_i)
        if type(self.grads['dE_drho_' + str(k) + '_' + str(i)]) is list:
            self.gradient['dE_drho_' + str(k) + '_' + str(i)] = dE_drho_k_i
            self.grads['dE_drho_' + str(k) + '_' + str(i)] = dE_drho_k_i
        else:
            self.gradient['dE_drho_' + str(k) + '_' + str(i)] += dE_drho_k_i
            self.grads['dE_drho_' + str(k) + '_' + str(i)] += dE_drho_k_i
            # output/input
        self.gradient['dz_' + str(k) + '_' + str(i) + '_dC_' + str(k) + '_' + str(i)] = dz_k_i_dC_k_i

        if node_name == 'node_k':
            self.gradient['dz_' + str(k) + '_' + str(i) + '_du_' + str(k) + '_' + str(i - 1)] = dz_k_i_du_k_i_1

    def node_k_backward(self, l):
        """
        Compute the forward pass for node k. Layer 7,11,25,29,43,47
        """
        self.node_k_l_backward(l)

    def node_k_forward(self, l):
        """
        Compute the forward pass for node k. Layer 7,11,25,29,43,47
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        rho_k_i = self.params['rho_' + str(k) + '_' + str(i)]
        C_k_i_1 = self.state_var['C_' + str(k) + '_' + str(i - 1)]
        u_k_i_1 = self.state_var['u_' + str(k) + '_' + str(i - 1)]
        z_k_i = np.zeros(C_k_i_1.shape)
        for m, C_k_i_m in enumerate(C_k_i_1):
            if C_k_i_m > 1 / rho_k_i:
                z_k_i[m] = C_k_i_m + u_k_i_1[m] - 1 / rho_k_i
            elif abs(C_k_i_m) < rho_k_i:
                z_k_i[m] = 0
            else:
                z_k_i[m] = C_k_i_m + u_k_i_1[m] + 1 / rho_k_i
        self.state_var['z_' + str(k) + '_' + str(i)] = z_k_i

    def node_i_backward(self, l):
        """
        Compute the backward pass for node i. Layer 5,9
        """
        self.node_e_f_g_i_j_h_backward(l)

    def node_i_forward(self, l):
        """
        Compute the forward pass for node i. Layer 5,9
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        Phi_bar = self.Phi_bar
        A_0 = 0.5 * Phi_bar
        rho_k_i = self.params['rho_' + str(k) + '_' + str(i)]
        F_k_i = self.params['F_' + str(k) + '_' + str(i)]
        z_k_i_1 = self.state_var['z_' + str(k) + '_' + str(i - 1)]
        u_k_i_1 = self.state_var['u_' + str(k) + '_' + str(i - 1)]

        temp1 = np.multiply(rho_k_i, np.dot(np.transpose(F_k_i), F_k_i))
        Omega_k_i_1 = np.linalg.inv(np.dot(np.transpose(A_0), A_0) + temp1)

        temp2 = np.multiply(rho_k_i, np.dot(np.transpose(F_k_i), (z_k_i_1 - u_k_i_1)))
        temp3 = np.sum(np.transpose(A_0), axis=1)
        Omega_k_i_2 = temp3 + temp2
        Omega_k_i = np.dot(Omega_k_i_1, Omega_k_i_2)
        self.state_var['Omega_' + str(k) + '_' + str(i)] = Omega_k_i

    def node_t_backward(self, l):
        """
        Compute the backward pass for node l. Layer 4,22,40
        """
        self.node_r_q_t_backward(l)

    def node_t_forward(self, l):
        """
        Compute the forward pass for node l. Layer 4,22,40
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        C_k_i = self.state_var['C_' + str(k) + '_' + str(i)]
        z_k_i = self.state_var['z_' + str(k) + '_' + str(i)]
        u_k_i = C_k_i - z_k_i
        self.state_var['u_' + str(k) + '_' + str(i)] = u_k_i

    def node_l_backward(self, l):
        """
        Compute the forward pass for node l. Layer Layer 3,21,39
        """
        self.node_k_l_backward(l)

    def node_l_forward(self, l):
        """
        Compute the forward pass for node l. Layer 3,21,39
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        rho_k_i = self.params['rho_' + str(k) + '_' + str(i)]
        C_k_i = self.state_var['C_' + str(k) + '_' + str(i)]
        z_k_i = np.zeros(C_k_i.shape)

        for m, C_k_i_m in enumerate(C_k_i):
            if C_k_i_m > 1 / rho_k_i:
                z_k_i[m] = C_k_i_m - 1 / rho_k_i
            elif abs(C_k_i_m) < rho_k_i:
                z_k_i[m] = 0
            else:
                z_k_i[m] = C_k_i_m + 1 / rho_k_i
        self.state_var['z_' + str(k) + '_' + str(i)] = z_k_i

    def node_m_backward(self, l):
        """
        Compute the backward pass for node m. Layer 2,6,10,20,24,28,38,42,46
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        M = self.M
        # loss/output
        dE_du_k_i = self.gradient['dE_du_' + str(k) + '_' + str(i)]
        dE_dz_k_i = self.gradient['dE_dz_' + str(k) + '_' + str(i)]
        dz_k_i_dC_k_i = self.gradient['dz_' + str(k) + '_' + str(i) + '_dC_' + str(k) + '_' + str(i)]
        du_k_i_dC_k_i = self.gradient['du_' + str(k) + '_' + str(i) + '_dC_' + str(k) + '_' + str(i)]
        dE_dC_k_i = np.dot(dE_du_k_i, du_k_i_dC_k_i) + np.dot(dE_dz_k_i, dz_k_i_dC_k_i)
        self.gradient['dE_dC_' + str(k) + '_' + str(i)] = dE_dC_k_i
        self.grads['dE_dC_' + str(k) + '_' + str(i)] = dE_dC_k_i
        # output/parameter
        Omega_k_i = self.state_var['Omega_' + str(k) + '_' + str(i)]
        dC_k_i_dF_k_i = np.zeros((M, M))
        dE_dF_k_i = np.zeros((M, M))
        for m in range(M):
            I_bar = np.zeros((M, M))
            I_bar[m, m] = 1
            d_n = np.dot(I_bar, Omega_k_i)
            dC_k_i_dF_k_i[:, [m]] = d_n  # column vector
            dE_dF_k_i[m, m] = np.dot(dE_dC_k_i, d_n)

        self.gradient['dC_' + str(k) + '_' + str(i) + '_dF_' + str(k) + '_' + str(i)] = dC_k_i_dF_k_i

        # loss/parameter
        if type(self.grads['dE_dF_' + str(k) + '_' + str(i)]) is list:
            self.gradient['dE_dF_' + str(k) + '_' + str(i)] = dE_dF_k_i
            self.grads['dE_dF_' + str(k) + '_' + str(i)] = dE_dF_k_i
        else:
            self.gradient['dE_dF_' + str(k) + '_' + str(i)] += dE_dF_k_i
            self.grads['dE_dF_' + str(k) + '_' + str(i)] += dE_dF_k_i
        # output/input
        F_k_i = self.params['F_' + str(k) + '_' + str(i)]
        self.gradient['dC_' + str(k) + '_' + str(i) + '_dOmega_' + str(k) + '_' + str(i)] = F_k_i  # dC_k_i_dOmega_k_i

    def node_m_forward(self, l):
        """
        Compute the forward pass for node m. Layer 2,6,10,20,24,28,38,42,46
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])
        F_k_i = self.params['F_' + str(k) + '_' + str(i)]
        Omega_k_i = self.state_var['Omega_' + str(k) + '_' + str(i)]
        C_k_i = np.dot(F_k_i, Omega_k_i)
        self.state_var['C_' + str(k) + '_' + str(i)] = C_k_i

    def node_h_forward(self, l):
        """
        Compute the forward pass for node h. Layer 1
        """
        Phi_bar = self.Phi_bar
        F_1_1 = self.params['F_1_1']
        rho_1_1 = self.params['rho_1_1']

        Phi_bar2 = np.dot(np.transpose(Phi_bar), Phi_bar)
        F_1_1_prev2 = np.dot(np.transpose(F_1_1), F_1_1)

        temp0 = np.multiply(rho_1_1, F_1_1_prev2)
        temp1 = np.linalg.inv(0.25 * Phi_bar2 + temp0)
        Omega_1_1 = np.sum(0.5 * np.dot(temp1, np.transpose(Phi_bar)), axis=1)

        self.state_var['Omega_1_1'] = Omega_1_1

    def node_h_backward(self, l):
        """
        Compute the backward pass for node h. Layer 1
        """
        self.node_e_f_g_i_j_h_backward(l)

    def node_e_f_g_i_j_h_backward(self, l):
        """
        Compute the backward pass for node f, e, g, i, j, h: common part
        """
        k = int(self.node_kvalue[l])
        i = int(self.node_ivalue[l])

        M = self.M
        N = self.N
        Layer_name = 'Layer' + str(l + 1)
        node_name = self.net_struct[Layer_name]

        if node_name == 'node_g' or node_name == 'node_h':
            z_k_i_1 = np.zeros((M, 1))
            u_k_i_1 = np.zeros((M, 1))
        elif node_name == 'node_f' or node_name == 'node_e' or node_name == 'node_i' or node_name == 'node_j':
            z_k_i_1 = self.state_var['z_' + str(k) + '_' + str(i - 1)]
            u_k_i_1 = self.state_var['u_' + str(k) + '_' + str(i - 1)]

        if node_name == 'node_i' or node_name == 'node_j' or node_name == 'node_h':
            Phi_bar = self.Phi_bar
            A_k_1 = 0.5 * Phi_bar
            b_k_1 = np.ones((N, 1))
        elif node_name == 'node_f' or node_name == 'node_e' or node_name == 'node_g':
            A_k_1 = self.state_var['A_' + str(k - 1)]
            b_k_1 = self.state_var['b_' + str(k - 1)]

        F_k_i = self.params['F_' + str(k) + '_' + str(i)]
        rho_k_i = self.params['rho_' + str(k) + '_' + str(i)]

        # loss / output

        if node_name == 'node_e' or node_name == 'node_g' or node_name == 'node_i' or node_name == 'node_h':
            dE_dC_k_i = self.gradient['dE_dC_' + str(k) + '_' + str(i)]
            dC_k_i_dOmega_k_i = self.gradient['dC_' + str(k) + '_' + str(i) + '_dOmega_' + str(k) + '_' + str(i)]
            dE_dOmega_k_i = np.dot(dE_dC_k_i, dC_k_i_dOmega_k_i)
        elif node_name == 'node_f' or node_name == 'node_j':
            dE_dBelta_k = self.gradient['dE_dBelta_' + str(k)]
            dBelta_k_dOmega_k_i = self.gradient['dBelta_' + str(k) + '_dOmega_' + str(k) + '_' + str(i)]
            dE_dOmega_k_i = np.dot(dE_dBelta_k, dBelta_k_dOmega_k_i)
        self.gradient['dE_dOmega_' + str(k) + '_' + str(i)] = dE_dOmega_k_i
        self.grads['dE_dOmega_' + str(k) + '_' + str(i)] = dE_dOmega_k_i
        # loss / parameter
        temp1 = np.multiply(rho_k_i, np.dot(F_k_i.T, F_k_i))
        Q = np.linalg.inv(np.dot(A_k_1.T, A_k_1) + temp1)
        Q3 = z_k_i_1 - u_k_i_1
        Q2 = np.dot(A_k_1.T, b_k_1) + np.multiply(rho_k_i, np.dot(F_k_i.T, Q3))
        Qs = np.dot(Q, Q)
        dOmega_k_i_drho_k_i = -np.dot(np.dot(np.dot(Qs, F_k_i.T), F_k_i), Q2) + np.dot(np.dot(Q, F_k_i.T), Q3)

        dOmega_k_i_dF_k_i = np.zeros((M, M))
        dE_dF_k_i = np.zeros((M, M))
        F1 = np.multiply(-2 * rho_k_i, Qs)
        for m in range(M):
            F_bar = np.zeros((M, M))
            I_bar = np.zeros((M, M))
            F_bar[m, m] = F_k_i[m, m]
            I_bar[m, m] = 1
            d_n = np.dot(np.dot(F1, F_bar), Q2) + np.dot(np.dot(np.multiply(rho_k_i, Q), I_bar), Q3)
            # print(d_n.shape)
            dOmega_k_i_dF_k_i[:, [m]] = d_n  # column vector
            dE_dF_k_i[m, m] = np.dot(dE_dOmega_k_i, d_n)

        self.gradient['dOmega_' + str(k) + '_' + str(i) + '_drho_' + str(k) + '_' + str(i)] = dOmega_k_i_drho_k_i
        self.gradient['dOmega_' + str(k) + '_' + str(i) + '_dF_' + str(k) + '_' + str(i)] = dOmega_k_i_dF_k_i

        dE_drho_k_i = np.dot(dE_dOmega_k_i, dOmega_k_i_drho_k_i)
        # self.gradient['dE_drho_'+str(k)+'_'+str(i)] = dE_drho_k_i
        # self.gradient['dE_dF_'+str(k)+'_'+str(i)] = dE_dF_k_i
        # self.grads['dE_drho_'+str(k)+'_'+str(i)] = dE_drho_k_i
        # self.grads['dE_dF_'+str(k)+'_'+str(i)] = dE_dF_k_i
        # loss/parameter
        if type(self.grads['dE_drho_' + str(k) + '_' + str(i)]) is list:
            self.gradient['dE_drho_' + str(k) + '_' + str(i)] = dE_drho_k_i
            self.grads['dE_drho_' + str(k) + '_' + str(i)] = dE_drho_k_i
        else:
            self.gradient['dE_drho_' + str(k) + '_' + str(i)] += dE_drho_k_i
            self.grads['dE_drho_' + str(k) + '_' + str(i)] += dE_drho_k_i

        if type(self.grads['dE_dF_' + str(k) + '_' + str(i)]) is list:
            self.gradient['dE_dF_' + str(k) + '_' + str(i)] = dE_dF_k_i
            self.grads['dE_dF_' + str(k) + '_' + str(i)] = dE_dF_k_i
        else:
            self.gradient['dE_dF_' + str(k) + '_' + str(i)] += dE_dF_k_i
            self.grads['dE_dF_' + str(k) + '_' + str(i)] += dE_dF_k_i

        # output / input
        if node_name != 'node_h':
            if node_name == 'node_f' or node_name == 'node_e' or node_name == 'node_i' or node_name == 'node_j':
                dOmega_k_i_dz_k_i_1 = np.multiply(rho_k_i, np.dot(Q, F_k_i.T))
                dOmega_k_i_du_k_i_1 = np.multiply(-rho_k_i, np.dot(Q, F_k_i.T))
                self.gradient[
                    'dOmega_' + str(k) + '_' + str(i) + '_dz_' + str(k) + '_' + str(i - 1)] = dOmega_k_i_dz_k_i_1
                self.gradient[
                    'dOmega_' + str(k) + '_' + str(i) + '_du_' + str(k) + '_' + str(i - 1)] = dOmega_k_i_du_k_i_1
            if node_name == 'node_f' or node_name == 'node_e' or node_name == 'node_g':
                dOmega_k_i_db_k_1 = np.dot(Q, A_k_1.T)
                self.gradient['dOmega_' + str(k) + '_' + str(i) + '_db_' + str(k - 1)] = dOmega_k_i_db_k_1
                dOmega_k_i_dA_k_1 = np.zeros((M, M, N))
                for n_x in range(N):
                    for n_y in range(M):
                        A_bar = np.zeros((M, M))
                        A_bar[n_y, :] = A_k_1[n_x, :]
                        A_bar2 = A_bar + A_bar.T
                        T_bar_b = np.zeros((M, 1))
                        T_bar_b[n_y] = b_k_1[n_x]
                        t_n = -np.dot(np.dot(Qs, A_bar2), Q2) + np.dot(Q, T_bar_b)
                        dOmega_k_i_dA_k_1[:, [n_y], [n_x]] = t_n
                self.gradient['dOmega_' + str(k) + '_' + str(i) + '_dA_' + str(k - 1)] = dOmega_k_i_dA_k_1

                ######################################################