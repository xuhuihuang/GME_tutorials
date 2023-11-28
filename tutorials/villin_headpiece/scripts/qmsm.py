import numpy as np

def preprocess(data, onlyone=False):
    if onlyone:
        if np.ndim(data)==1: 
            data = data.reshape(1, len(data[0]))
        elif np.ndim(data)==2:
            data = data.reshape(1, len(data[0]), len(data[1]))
    if (np.ndim(data)==2):
        dim = int(np.sqrt(len(data[0])))
    elif (np.ndim(data)==3) :
        if (len(data[0]) != len(data[0][0])) :
            dim = int(np.sqrt(len(data[0]) * len(data[0][0])))
        else :
            dim = len(data[0])
    else :
        return 0
    reshaped_data = data.reshape([len(data), dim, dim])
    if abs((np.sum(reshaped_data[0, 0])) - 1) > 1e-3 and \
        abs(np.sum(reshaped_data[0, :, 0]) - 1) > 1e-3:
        return 0
    if abs((np.sum(reshaped_data[0, 0])) - 1) > 1e-3 and \
        abs(np.sum(reshaped_data[0, :, 0]) - 1) < 1e-3:
        for i in range(len(reshaped_data)):
            reshaped_data[i] = reshaped_data[i].T
    return dim, reshaped_data

def compute_sp(data):
    dim, input_TPM = preprocess(data)
    density = np.zeros((len(input_TPM), dim))
    for i in range(len(input_TPM)):
        ev, vr = np.linalg.eig(input_TPM[i].T)
        sorted_indices = np.argsort(ev)
        density[i] = vr[:,sorted_indices[-1]] / np.sum(vr[:,sorted_indices[-1]])
    return np.mean(density, axis=0)

def compute_ITS(lag, TPM):
    dim = len(TPM)
    ev, _ = np.linalg.eig(TPM.T)
    ev = sorted(ev, reverse=True)
    ITS = np.zeros(dim-1)
    for i in range(len(ITS)):
        ITS[i] = -lag / np.log(np.real(ev[i+1]))
    return ITS
        
class QuasiMSM(object):
    """
    
    ----------
    
    """
    def __init__(self):
        self._K_matrix = []
        self._dTPM_dt_0 = []
        self._sp = []
        self._mik = []
        self._rmse = []
        self._dim = 0
        self._tau_k = 0
        self._delta_t = 0
    
    @property
    def K(self):
        return self._K_matrix
    
    @property
    def mik(self):
        return self._mik
        
    @property
    def rmse(self):
        return self._rmse

    @property
    def sp(self):
        return self._sp
    
    def fit(self, data, tau_k, delta_t=1, add_iden_mat=False, rmse=True, rmse_weighted_by_sp=True):
        """
        rmse_range will have a little difference between add_iden_mat or not
        rmse_range is a list

        Returns
        -------
  
        """  
        self._dim, input_TPM = preprocess(data)
        if np.ndim(input_TPM) != 3:
            raise IOError("Transition Probablity Matrix is not valid to do qMSM")
        if add_iden_mat:
            input_TPM = np.insert(input_TPM, 0, np.eye(self._dim), axis=0)
        self._delta_t = delta_t
        self._tau_k = tau_k
        self._K_matrix = self._compute_K_matrix(input_TPM, add_iden_mat=add_iden_mat)
        self._sp = compute_sp(input_TPM[int(self._tau_k/2) : self._tau_k+2])
        for i in range(self._tau_k):
            integral_kernel = np.sum(self._K_matrix[:i+1], axis=0) * self._delta_t
            self._mik.append(np.linalg.norm(integral_kernel)/self._dim)
            if rmse:
                self._rmse.append(self._compute_qmsm_rmse(input_TPM, tau_k = i+1, 
                                                          rmse_weighted_by_sp = rmse_weighted_by_sp))
                        
    def predict(self, data, tau_k=None, begin=1, end=None, add_iden_mat=False, ITS=False):
        if tau_k is None:
            tau_k = self._tau_k
        if end is None:
            end = len(data)
        if (tau_k > self._tau_k) or (len(data) < tau_k+2):
            return 0
        TPM_time = np.arange(begin, end+1) * self._delta_t
        raw_TPM = preprocess(data)[1][:tau_k+2]
        pre_TPM = np.zeros(((max(end, self._tau_k+2), self._dim, self._dim)))              
        pre_TPM[:tau_k+2] = raw_TPM
        for i in range(tau_k+2, len(pre_TPM)):
            memory_term = np.zeros((self._dim, self._dim))
            for k in range(tau_k):
                memory_term += pre_TPM[i-k-2] @ self._K_matrix[k]
            dTPM = pre_TPM[i-1] @ self._dTPM_dt_0 - memory_term * self._delta_t
            pre_TPM[i] = pre_TPM[i-1] + dTPM * self._delta_t
        pre_TPM = pre_TPM[begin-1 : end]
        if ITS:
            ITS_mat = np.zeros((len(TPM_time), self._dim-1))
            for i in range(len(TPM_time)):
                ITS_mat[i] = compute_ITS(TPM_time[i], pre_TPM[i])
        if add_iden_mat:
            TPM_time = np.insert(TPM_time, 0, 0)
            pre_TPM = np.insert(pre_TPM, 0, np.eye(self._dim), 0)
            ITS_mat = np.insert(ITS_mat, 0, -1*np.ones(self._dim-1), 0)
        if ITS:
            return TPM_time, pre_TPM, ITS_mat
        return TPM_time, pre_TPM
    
    def timescales(self, data, ITS_t, tau_k=None):
        if tau_k is None:
            tau_k = self._tau_k
        TPM_time, pre_TPM = self.predict(data, tau_k, end=ITS_t)
        return compute_ITS(TPM_time[-1], pre_TPM[-1])
        
    def _compute_dTPM_dt(self, input_data, add_iden_mat=False):
        dTPM_dt = []
        for i in range(1, self._tau_k+2):
            dTPM_dt.append((input_data[i] - input_data[i-1]) / self._delta_t)
        if add_iden_mat:
            dTPM_dt_0 = np.linalg.inv((input_data[0] + input_data[1]) / 2) @ dTPM_dt[0]
        else:
            dTPM_dt_0 = np.linalg.inv(input_data[0]) @ dTPM_dt[0]
        return dTPM_dt_0, dTPM_dt
        
    def _compute_K_matrix(self, input_data, add_iden_mat=False):
        """
        ----------
        Returns
        -------
        K_matrix: Memory kernel calculation result tensor with cal_step entries.
        """
        K_matrix = np.zeros((self._tau_k, self._dim, self._dim))
        self._dTPM_dt_0, dTPM_dt = self._compute_dTPM_dt(input_data, add_iden_mat=add_iden_mat)
        K_matrix[0] = (np.linalg.inv(input_data[0]) @ \
                       (dTPM_dt[1] -  input_data[1] @ self._dTPM_dt_0)) / self._delta_t
        if self._tau_k == 1:
            return -1 * K_matrix
        for n in range(1, self._tau_k):
            memory_term = np.zeros((self._dim, self._dim))
            for m in range(0, n):
                memory_term += input_data[n-m] @ K_matrix[m]
            K_matrix[n] = np.linalg.inv(input_data[0]) @ \
                ((dTPM_dt[n+1] - input_data[n+1] @ self._dTPM_dt_0) / self._delta_t - memory_term)
        return -1 * K_matrix
        
    def _compute_qmsm_rmse(self, input_data, tau_k, rmse_weighted_by_sp=True):
        _, predicted_data = self.predict(input_data, tau_k, end=len(input_data))
        diff = predicted_data - input_data
        if (rmse_weighted_by_sp):
            for i in range(len(diff)) :
                diff[i] = np.diag(self._sp) @ diff[i]
        return np.sqrt(np.mean(np.square(diff)))
        
    def top_model(self, data, ini=1, add_iden_mat=False):
        if len(self._rmse) == 0:
            return 0
        _, input_TPM = preprocess(data)
        tauk_idx = np.argmin(self._rmse[ini-1:])
        mdl = QuasiMSM()
        mdl.fit(input_TPM, tauk_idx+ini, delta_t=self._delta_t, add_iden_mat=add_iden_mat, rmse=False)
        mdl._rmse = self._rmse[tauk_idx+ini-1]
        return mdl
    
class MSM(object):
    """
    
    ----------
    
    """
    def __init__(self):
        self._TPM = []
        self._sp = []
        self._timescales = []
        self._rmse = []
        self._dim = 0
        self._tau = 0
        self._delta_t = 0
    
    @property
    def sp(self):
        return self._sp
    
    @property
    def rmse(self):
        return self._rmse
    
    @property
    def timescales(self):
        return self._timescales
    
    def fit(self, data, tau=1, delta_t = 1, rmse=False, rmse_weighted_by_sp=True):
        self._dim, raw_TPM = preprocess(data)
        self._tau = tau
        self._delta_t = delta_t
        self._TPM= raw_TPM[tau-1]
        self._sp = compute_sp(raw_TPM)
        self._timescales = compute_ITS(self._tau * self._delta_t, self._TPM)
        if rmse:
            end = len(raw_TPM)
            _, predicted_TPM = self.predict(end=end)
            diff = np.zeros((int(end//self._tau), self._dim, self._dim))
            for i in range(len(diff)):
                diff[i] = predicted_TPM[i] - raw_TPM[(i+1)*self._tau-1]
                if rmse_weighted_by_sp:
                    diff[i] = np.diag(self._sp) @ diff[i]
            self._rmse = np.sqrt(np.mean(np.square(diff))) 
    
    def predict(self, end=100, add_iden_mat=False):
        TPM_time = np.arange(1, int(end//self._tau)+1) * self._tau * self._delta_t
        predicted_TPM = np.zeros((int(end//self._tau), self._dim, self._dim))
        for i in range(len(predicted_TPM)):
            predicted_TPM[i] = np.linalg.matrix_power(self._TPM, i+1)
        if add_iden_mat:
            TPM_time = np.insert(TPM_time, 0, 0)
            predicted_TPM = np.insert(predicted_TPM, 0, np.eye(self._dim), 0)
        return TPM_time, predicted_TPM
    
    def scan(self, data, tau_lst, delta_t = 1, rmse_weighted_by_sp=True):
        rmse = []
        for tau in tau_lst:
            temp_mdl = MSM()
            temp_mdl.fit(data, tau=tau, delta_t = delta_t, rmse=True, rmse_weighted_by_sp=rmse_weighted_by_sp)
            rmse.append(temp_mdl.rmse)
        return rmse

