# Author: Yunrui Qiu <yunruiqiu@gmail.com>
# Contributors: Bojun Liu <bliu293@wisc.edu>; Yue Wu <wu678@wisc.edu>
# The implementation of VAMPNet models is inspired by deeptime
# Copyright (c) 2023, Unveristy of Wisconsin-Madison
# All rights reserved.


import numpy as np
import torch
import os
from tqdm import trange, tqdm
import scipy.linalg
from copy import deepcopy
import random


def TimeLaggedDataset(trajs=None, lagtime=100, normalize=False, mean=None, std=None):

    """
    create the time lagged dataset from the featurized trajectories

    Parameters
    ----------
    trajs: list-like, shape: (num_trajs, )(length_traj, num_features)
        The ensemble of trajectories used to create time lagged dataset.
    lagtime: int, default=100
        The number of frames used as the lagtime.
    normalize: bool, default=False
        If the pastdata and futuredata are normalized respectively.
    mean: float, default=None
        The mean value used to normalize the data, if not given, the mean value of input data will be used.
    std: float, default=None
        The standard deviation used to normalize the data, if not given, the std of the input data will be used.

    Returns
    ----------
    _pastdata, futuredata: torch.Tensor, shape: (num_trajs*length_traj-num_trajs*lagtime, num_features)
        The time lagged dataset.
    """

    _pastdata = None; _futuredata = None
    for i in tqdm(range(len(trajs)), desc='load data'):
        for j in range(len(trajs[i])-int(lagtime)):
            if _pastdata != None and _futuredata != None:
                _pastdata.append(trajs[i][j])
                _futuredata.append(trajs[i][j+lagtime])
            else:
                _pastdata = [trajs[i][j]]
                _futuredata = [trajs[i][int(j+lagtime)]]
    _pastdata = np.array(_pastdata); _futuredata = np.array(_futuredata)
    if  normalize:
        if not mean == None and not std == None:
            past_mean = mean; future_mean = mean 
            past_std = std; future_std = std
        else:
            past_mean = np.mean(_pastdata, axis=0); future_mean = np.mean(_futuredata, axis=0)
            past_std = np.std(_pastdata, axis=0); future_std = np.std(_futuredata, axis=0)
        _pastdata -= past_mean[None, :]; _pastdata /= past_std[None, :]
        _futuredata -= future_mean[None, :]; _futuredata /= future_std[None, :]
    _pastdata.astype(np.float32); _futuredata.astype(np.float32)
    _pastdata = torch.from_numpy(_pastdata); _futuredata = torch.from_numpy(_futuredata)
    return _pastdata, _futuredata


def split_train_validate_data(pastdata, futuredata, validation_ratio=0.1, train_batchsize=10000):

    """
    Split the time lagged dataset into training and validation dataset and create the dataloader.

    Parameters
    ----------
    pastdata: torch.Tensor, shape: (num_trajs*length_traj-num_trajs*lagtime, num_features)
        The pastdata from the TimeLaggedDataset.
    futuredata torch.Tensor, shape: (num_trajs*length_traj-num_trajs*lagtime, num_features)
        The futuredata from the TimeLaggedDataset.
    validation_ratio: float, default=0.1
        The ratio of the validation dataset (should be in range(0, 1)).
    train_batchsize: int, default=10000
        The single batchsize of the training data loader.

    Returns
    ----------
    train_loader, validate_loader: torch.utils.data.dataloader.Dataloader
        The training and validation dataloader.
    
    """

    assert pastdata.shape == futuredata.shape, "The pastdata and futuredata should share the same shape."
    dataset = []
    for i in range(len(pastdata)):
        dataset.append((pastdata[i], futuredata[i]))
    train_set, validate_set = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)*validation_ratio), int(len(dataset)*validation_ratio)])
    train_loader = torch.utils.data.dataloader.DataLoader(train_set, batch_size=train_batchsize, shuffle=False)
    validate_loader = torch.utils.data.dataloader.DataLoader(validate_set, batch_size=int(len(validate_set)*0.2), shuffle=False)
    return train_loader, validate_loader


def rao_blackwell_ledoit_wolf(covmatrix=None, num_data=None):
    """
    Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance matrix.
    [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
    estimation of high dimensional covariance matrices" ICASSP (2009)

    Parameters
    ----------
    covmatrix: torch.Tensor, default=None
        The covariance matrix used to perform the Rao-Blackwellized Ledoit-Wolf shrinkaged estimation.
    num_data: int, default=None
        The number of the data used to compute the covariance matrix.
    
    ----------

    """
    matrix_dim = covmatrix.shape[0]
    assert covmatrix.shape == (matrix_dim, matrix_dim), "The input covariance matrix does not have the squared shape;"

    alpha = (num_data-2)/(num_data*(num_data+2))
    beta = ((matrix_dim+1)*num_data - 2) / (num_data*(num_data+2))

    trace_covmatrix_squared = torch.sum(covmatrix*covmatrix)  
    U = ((matrix_dim * trace_covmatrix_squared / torch.trace(covmatrix)**2) - 1)
    rho = min(alpha + beta/U, 1)

    F = (torch.trace(covmatrix) / matrix_dim) * torch.eye(matrix_dim).to(device=covmatrix.device)
    return (1-rho)*covmatrix + rho*F, rho


def covariance(pastdata=None, futuredata=None, remove_mean=True):

    """
    Calculation of self, instantaneous, time-lagged correlation matrix. C_{00}, c_{0t}, c_{tt}

    Parameters
    ----------
    pastdata: torch.Tensor, default=None, shape:(length_data, num_features)
        Pastdata used to calculate time=0 correlation. Can be generated from function: TimeLaggedDataset.
    futuredata: torch.Tensor, default=None, shape:(length_data, num_features)
        Futuredata used to calculate time=t correlation. Can be generated from function: TimeLaggedDataset.
    remove_mean: bool, default:True
        The bool value used to decide if to remove the mean values for both the pastdata and futuredata.

    Returns
    ----------
    c00: torch.Tensor, shape: (num_features, num_features) 
        Self-instantaneous correlation matrix generated from pastdata.
    c0t: torch.Tensor, shape: (num_features, num_features)
        Time lagged correlation matrix generated from pastdata.
    ctt: torch.Tensor, shape: (num_features, num_features)
        Self-instantaneous correlation matrix generated from futuredata. 
    """

    assert pastdata.shape == futuredata.shape, "pastdata and future mush share the same shape"
    if remove_mean:
        pastdata = pastdata - torch.mean(pastdata, dim=0)
        futuredata = futuredata - torch.mean(futuredata, dim=0)
    
    c00 = torch.matmul(pastdata.t(), pastdata) / (pastdata.shape[0] - 1)
    ctt = torch.matmul(futuredata.t(), futuredata) / (futuredata.shape[0] -1)
    c0t = torch.matmul(pastdata.t(), futuredata) / (pastdata.shape[0] - 1)
    
    c00, _ = rao_blackwell_ledoit_wolf(covmatrix=c00, num_data=pastdata.shape[0])
    ctt, _ = rao_blackwell_ledoit_wolf(covmatrix=ctt, num_data=futuredata.shape[0])
    return c00, c0t, ctt


def matrix_decomposition(matrix=None, epsilon=1e-6, method='regularize'):

    """
    Eigen-decomposition for the input hermetian matrix.

    Parameters
    ----------
    matrix: torch.Tensor, default=None
        The hermetian matrix used to perform eigen-decomposition.
    epsilon: float, default=1e-6
        The numerical modification put on the matrix to ensure its positive rank and may used as cutoff for eigenvalues.
    method: string, default='regularize
        Use three different methods to do eigen-decomposition: 'regularize', 'cutoff', 'clamp'
        regularize: Modify the input matrix by adding epsilon driven identify matrix and take absolute value for eigenvalues;
        cutoff: Set epsilon as the cutoff for the eigenvalues of the matrix;
        clamp: Set all eigenvalues smaller than epsilon to be epsilon;
    
    Returns
    ----------
    eigenvalues, eigenvectors (torch.Tensor) of the input hermetian matrix.
    """

    methods = ['regularize', 'cutoff', 'clamp']
    assert method in methods, "Invalid method "+str(method)+", should use methods from "+str(methods)

    _matrix = matrix + torch.eye(matrix.shape[0]) * epsilon
    eigenval, eigenvec = torch.linalg.eigh(_matrix)
    eigenvec = eigenvec.t()

    if method == "regularize":
        eigenval = torch.abs(eigenval)
#         eigenvec = torch.abs(eigenvec)
    elif method == "cutoff":
        _mask = eigenval > epsilon
        eigenval = eigenval[_mask]
        eigenvec = eigenvec[_mask]
    elif method == "clamp":
        eigval = torch.clamp_min(eigval, min=epsilon)
    return eigenval, eigenvec


def matrix_inverse(matrix=None, sqrt_root=False, epsilon=1e-6, method='regularize'):

    """
    Calculate the inverse of the matrix.

    Parameters
    ----------
    matirx: torch.Tensor, default=None
        The matrix used to calculate the inverse.
    sqrt_root: bool, default=False
        The bool value to decide if the sqrt root inverse of the matrix is calculated.
    epsilon: float, default=1e-6
        The epsilon modification used to do eigen-decomposition of the input matrix.
    method: string, default='regularize'
        The method used to do eigen-decomposition for the input matrix: 'regularize', 'cutoff', 'clamp'

    Returns
    ---------
    _inv_matrix: torch.Tensor
        The inverse of the input matrix.
    """

    eigenval, eigenvec = matrix_decomposition(matrix=matrix, epsilon=epsilon, method=method)
    if not sqrt_root:
        _inv_matrix = torch.matmul(eigenvec.t(), torch.matmul(torch.diag(1 / eigenval), eigenvec))
    else:
        _inv_matrix = torch.matmul(eigenvec.t(), torch.matmul(torch.diag(1 / torch.sqrt(eigenval)), eigenvec))
    return _inv_matrix


def koopman_matrix(pastdata=None, futuredata=None, epsilon=1e-6, method='cutoff'):

    """
    Compute the koopman matrix for the time-lagged data based on the VAMP theory.
    koopman_matrix = c_{00}^{-1/2}c_{0t}c_{tt}^{-1/2}

    Parameters
    ----------
    pastdata: torch.Tensor, default=None
        The past part of the time-lagged sequence data. Can be generated from TimeLaggedDataset.
    futuredata: torch.Tensor, default=None
        The future part of the time-lagged sequence data. Can be generated from TimeLaggedDataset.
    epsilon: float, default=1e-6
        The epsilon modification used to calculate the inverse of the instantaneous matrix.
    method: string, default='cutoff'
        The method used to calculate the inverse of the instantaneous matix: : 'regularize', 'cutoff', 'clamp'
    
    Returns
    ----------
    _koopman_matrix: torch.Tensor, shape:(num_features, num_features)
        Koopman matrix for the input time-lagged sequence data.

    """

    c00, c0t, ctt = covariance(pastdata=pastdata, futuredata=futuredata, remove_mean=True)
    c00_sqrt_inv = matrix_inverse(matrix=c00, sqrt_root=True, epsilon=epsilon, method=method)
    ctt_sqrt_inv = matrix_inverse(matrix=ctt, sqrt_root=True, epsilon=epsilon, method=method)
    
    _koopman_matrix = torch.matmul(c00_sqrt_inv, torch.matmul(c0t, ctt_sqrt_inv)).t()
    return _koopman_matrix


def reverse_propogator(pastdata=None, futuredata=None, epsilon=1e-6, method='cutoff'):

    """
    Compute the reverse propogator for the time-lagged sequence data with detailed balance contraint.
    This function is used to calculate the loss function for SRVNet.
    cholesky decomposition of reversed instantaneous correlation matrix: LL^{T} = (c_{00} + c_{tt}}) / 2
    propogator = L ((c_{0t} + c_{t0}}) / 2) L^{T}

    Parameters
    ----------
    pastdata: torch.Tensor, default=None
        The past part of the time-lagged sequence data. Can be generated from TimeLaggedDataset.
    futuredata: torch.Tensor, default=None
        The future part of the time-lagged sequence data. Can be generated from TimeLaggedDataset.
    epsilon: float, default=1e-6
        The epsilon modification used to calculate the inverse of the instantaneous matrix.
    method: string, default='cutoff'
        The method used to calculate the inverse of the instantaneous matix: : 'regularize', 'cutoff', 'clamp'
    
    Returns
    ----------
    _reverse_propogator: torch.Tensor, shape:(num_features, num_features)
        Reverse propogator matrix for the input time-lagged sequence data.
    """

    c00, c0t, ctt = covariance(pastdata=pastdata, futuredata=futuredata, remove_mean=True)
    _, ct0, _ = covariance(pastdata=futuredata, futuredata=pastdata, remove_mean=True)
    
    c0 = (c00 + ctt) / 2
    ct = (c0t + ct0) / 2

    _lower_triangle = torch.linalg.cholesky(c0)
    _lower_triangle_inv = torch.inverse(_lower_triangle)
    _reverse_propogator = torch.matmul(_lower_triangle_inv, torch.matmul(ct, _lower_triangle_inv.t()))

    return _reverse_propogator


def loss_function(pastdata=None, futuredata=None, criterion='SRVNet', epsilon=1e-6, method='cutoff'):

    """
    Loss function calculation for the VAMPNet / SRVNet training.

    Parameters
    ----------
    pastdata: torch.Tensor, default=None
        The past part of the time-lagged sequence data. Can be generated from TimeLaggedDataset.
    futuredata: torch.Tensor, default=None
        The future part of the time-lagged sequence data. Can be generated from TimeLaggedDataset.
    criterion: string, default='SRVNet'
        The criterion used to calculate the score/loss fucntion, can be chosen from: 'SRVNet', 'VAMP1', 'VAMP2', 'VAMPE'
    epsilon: float, default=1e-6
        The epsilon modification used to calculate the inverse of the instantaneous matrix.
    method: string, default='cutoff'
        The method used to calculate the inverse of the instantaneous matix: : 'regularize', 'cutoff', 'clamp'
    
    Returns
    ----------
    loss function: float
        The loss function for the set criterion.
    """

    _criterion = ['SRVNet', 'VAMP1', 'VAMP2', 'VAMPE']
    assert criterion in _criterion, "Invalid criterion "+str(criterion)+" was adopted, should use criterion in: "+str(_criterion)
    
    if criterion == 'SRVNet':
        rev_propogator = reverse_propogator(pastdata=pastdata, futuredata=futuredata, epsilon=0, method='regularize')
        _eigenval, _ = torch.linalg.eigh(rev_propogator)
        _score = torch.sum(torch.pow(_eigenval, 2)) + 1


    elif criterion == 'VAMP1':
        koopman_mat = koopman_matrix(pastdata=pastdata, futuredata=futuredata, epsilon=epsilon, method=method)
        _score = torch.norm(koopman_mat, p='nuc') + 1


    elif criterion == 'VAMP2':
        koopman_mat = koopman_matrix(pastdata=pastdata, futuredata=futuredata, epsilon=epsilon, method=method)
        _score = torch.pow(torch.norm(koopman_mat, p='fro'), 2) + 1


    elif criterion == 'VAMPE':
        c00, c0t, ctt = covariance(pastdata=pastdata, futuredata=futuredata, remove_mean=True)
        c00_sqrt_inv = matrix_inverse(matrix=c00, sqrt_root=True, epsilon=epsilon, method=method)
        ctt_sqrt_inv = matrix_inverse(matrix=ctt, sqrt_root=True,  epsilon=epsilon, method=method)

        koopman_mat = torch.matmul(c00_sqrt_inv, torch.matmul(c0t, ctt_sqrt_inv)).t()
        u, sigma, v = torch.svd(koopman_mat)
        mask = sigma > epsilon
        
        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        sigma = sigma[mask]

        u_t = u.t(); v_t = v.t()
        sigma = torch.diag(sigma)
        _score = torch.trace(2. * torch.linalg.multi_dot([sigma, u_t, c0t, v]) - torch.linalg.multi_dot([sigma, u_t, c00, u, sigma, v_t, ctt, v])) + 1

    return -1. * _score


def map_data(data, device=None):

    """
    Data generator: map the input trajectories data to thee torch tensor data and change the dtype of data.

    Parameters
    ----------
    data: list, tuple like, (num_trajs, )(length_trajectory, num_features)
        Input data which will be transfered to torch.tensor. The default type for data is 32 floating point.
    device: device, default=None
        The device for the input data. Can be None which defaults to CPU.

    Returns
    ----------
    data generator: torch.Tensor
    """

    with torch.no_grad():
        if not isinstance(data, (list, tuple)):
            data = [data]
        else:
            for _data in data:
                if isinstance(_data, torch.Tensor):
                    _data = _data.to(device=device)
                else:
                    _data = torch.from_numpy(np.asarray(_data, dtype=np.float32).copy()).to(device=device)
                yield _data

                
def set_random_seed(random_seed):

    """
    Set the random seed for the torch, numpy, random functions
    """

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


class Network_Model(object):

    """
    A neural network model which can be fit to data optimizing for one of the implemented VAMP / SRVNet scores.

    Parameters
    ----------
    lobe : torch.nn.Module
        The past lobe of the VAMPNet / SRVNet. 
        The SRVNet shares the weight for past and future lobe, so only one lobe is trained.
        The VAMPNet can be used for irreversible dynamics, two different lobes will be trained. Or two lobes can also be shared.
    lobe_timelagged : torch.nn.Module, optional, default=None
        The timelagged lobe. Can be left None, in which case the lobes are shared.
    device : device, default=None
        The device for the lobe(s). Can be None which defaults to CPU.

    """

    def __init__(self, lobe, lobe_timelagged=None, device=None):

        self._lobe = lobe
        self._lobe_timelagged = lobe_timelagged if lobe_timelagged is not None else lobe
        self._device = device

    @property   
    def lobe(self):
        return self._lobe
    
    @property
    def lobe_timelagged(self):
        return self._lobe_timelagged
    
    def transform(self, data, instantaneous=True):

        if instantaneous:
            self._lobe.eval()
            network = self._lobe
        else:
            self._lobe_timelagged.eval()
            network = self._lobe_timelagged

        output = []
        for _data in map_data(data=data, device=self._device):
            output.append(network(_data).detach().cpu().numpy())
        return output if len(output) > 1 else output[0]


class deep_projector(object):

    """
    A neural network based class used to do dimensionality reduction. SRVNet and VAMPNet with different types of VAMP score
    are included in the class.
    [1]. Mardt, A., Pasquali, L., Wu, H. et al. VAMPnets for deep learning of molecular kinetics. Nat Commun 9, 5 (2018).
    [2]. Chen, W. et al. Nonlinear discovery of slow molecular modes using state-free reversible VAMPnets. J. Chem. Phys. 150, 214114 (2019) 

    Parameters
    ----------
    network_type: string, default='SRVNet'
        The type of the neural network and the corresponding loss function, can be chosen from: 'SRVNet', 'VAMP1', 'VAMP2', 'VAMPE';
    lobe: torch.nn.Module, default=None
        The architecture of the network.
        The SRVNet shares the weight for past and future lobe, so only one lobe is trained.
        The VAMPNet can be used for irreversible dynamics, two different lobes will be trained. Or two lobes can also be shared.
    lobe_timelagged : torch.nn.Module, optional, default=None
        The timelagged lobe. Can be left None, in which case the lobes are shared.
    epsilon: float, default=1e-6
        The cutoff / modification number for matrix eigen-decomposition calculation. See details in above functions.
    covaraince_method: string, default='regularize'
        The method used for matrix eigen-decomposition calculation. See details in above functions.
    device : device, default=torch.device("cpu")
        The device for the lobe(s). Can be None which defaults to CPU.
    learning_rate: float, default=1e-4
        The learning rate used to weight the gradient to update the network
    optimizer: float, default: Adam without weight decay
        The optimization algorithm used to update the network
    """

    def __init__(self, network_type='SRVNet', lobe=None, lobe_timelagged=None, epsilon=1e-6, 
                 covariance_method='regularize', device=torch.device("cpu"), learning_rate=1e-4, optimizer=None):

        self._network_type = network_type
        self._lobe = lobe
        self._lobe_timelagged = lobe_timelagged
        self._device = device
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self.__epsilon = epsilon
        self.__covariance_method = covariance_method
        self.__step = 0
        
        networktype_list = ['SRVNet', 'VAMP1', 'VAMP2', 'VAMPE']
        assert self._network_type in networktype_list, "Invalid network type "+str(self._network_type)+", should adopt from "+str(networktype_list)
        assert self._lobe != None, "Input argument {lobe} cannot be None."
        if optimizer == None and self._lobe_timelagged == None:
            self._optimizer = torch.optim.Adam(self._lobe.parameters(), lr=self._learning_rate)
        elif optimizer == None and self._lobe_timelagged != None:
            self._optimizer= torch.optim.Adam(list(self._lobe.parameters()) + list(self._lobe_timelagged.parameters()), lr=self._learning_rate)
        
        ### SRVNet should share the same weight for both instantaneous and timelagged lobe
        ### Eigenvalues for the reweighted time correlation matrix are also saved during the training of SRVNet
        ### so that it is clearly which mode is optimized during training;
        if self._network_type == 'SRVNet':
            self._lobe_timelagged = None
            self._srvnet_train_eigenvals = []
            self._srvnet_validate_eigenvals = []
            self._normal_mean = None
            self._normal_eigenvecs = None
        self._train_scores = []
        self._validation_score = []


    @property
    def lobe(self):
        """ The instantaneous lobe of deep projector, type:torch.nn.Module.
        :getter: Gets the instantaneous lobe.
        :setter: Sets a new lobe.
        """
        return self._lobe
    

    @lobe.setter
    def lobe(self, network):

        self._lobe = network
        self._lobe = self._lobe.to(device=self._device)


    @property
    def optimizer(self):
        """
        The optimizer used to updated the neural network, type: torch.optim.
        :setter: Sets a new optimizer.
        """
        return self._optimizer
    

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt


    @property
    def lobe_timelagged(self):
        """ The time lagged lobe of deep projector, type:torch.nn.Module.
        :getter: Gets the instantaneous lobe.
        :setter: Sets a new lobe.
        """
        return self._lobe_timelagged
    

    @lobe_timelagged.setter
    def lobe_timelagged(self, network=None):
        if network == None:
            self._lobe_timelagged = self.lobe   
        else:
            self._lobe_timelagged = network
        self._lobe_timelagged = self._lobe_timelagged.to(device=self._device)

    
    @property
    def eigenvalues(self):
        """
        For the training of SRVNet, the eigenvalues of the reverse propgator can represent the timescales of transition modes.
        Redo normalization again after training the SRVNet.
        """

        if self._network_type == 'SRVNet':
            return self._srvnet_train_eigenvals, self._srvnet_validate_eigenvals
        else:
            raise InterruptedError("Cannot output the explicit eigenvalues based on VAMPNet model.")
    

    @property
    def train_score(self):
        return self._train_scores


    @property
    def validate_score(self):
        return self._validation_score
    

    @property
    def srv_normal_mean(self):
        if self._network_type != 'SRVNet':
            raise InterruptedError("VAMPNet based model does not need to do any normalization.")
        else:
            return self._normal_mean
        
    
    @property
    def srv_normal_eigenvecs(self):
        if self._network_type != 'SRVNet':
            raise InterruptedError("VAMPNet based model does not need to do any normalization.")
        else:
            return self._normal_eigenvecs
    


    def _normalize(self, dataloader):

        """
        Training SRVNet does not pose the orthogonality of the projected collective variables.
        Need redo the normalization for the neural network projection and then project on ordered orthogonal eigenvectors. 
        """

        if not self._network_type == "SRVNet":
            raise InterruptedError("VAMPNet based model does not need to do any normalization.")
        else:

            with torch.no_grad():
                __pastdata = None; __futuredata = None
                for __past_data, __future_data in dataloader:
                    if __pastdata != None and __futuredata != None:
                        __pastdata = torch.cat((__pastdata, self.lobe(__past_data)), dim=0)
                        __futuredata = torch.cat((__futuredata, self.lobe(__future_data)), dim=0)
                    else:
                        __pastdata = self.lobe(__past_data); __futuredata = self.lobe(__future_data)

                pastmean = torch.mean(__pastdata, dim=0); futuremean = torch.mean(__futuredata, dim=0)
                mean = ((pastmean + futuremean) / 2).detach().numpy()
                c00, c0t, ctt = covariance(pastdata=__pastdata, futuredata=__futuredata, remove_mean=True)
                _, ct0, ctt = covariance(pastdata=__futuredata, futuredata=__pastdata, remove_mean=True)

                c0 = (c00 + ctt) / 2
                ct = (c0t + ct0) / 2

                c0 = c0.numpy(); ct = ct.numpy()
                eigenval, eigenvecs = scipy.linalg.eigh(ct, b=c0)
                index = np.argsort(eigenval)[::-1]
                eigenval = eigenval[index]
                eigenvecs = eigenvecs[:,index]
        return mean, eigenvecs


    def partial_fit(self, pastdata, futuredata):

        """
        Train the projector model using one batchsize data.
        Will automatically exam the type of data and the type of the network.
        """
        
        assert pastdata.shape == futuredata.shape, "Input pastdata and futuredata should share the same shape: "+ str(pastdata.shape) + "!=" + str(futuredata.shape)
        self._lobe.train()
        if not self._network_type == 'SRVNet':
            if self._lobe_timelagged == None:
                self._lobe_timelagged = self._lobe
            self._lobe_timelagged.train()

        if isinstance(pastdata, np.ndarray):
            pastdata = torch.from_numpy(pastdata.astype(np.float32))
        if isinstance(futuredata, np.ndarray):
            pastdata = torch.from_numpy(futuredata.astype(np.float32))
        pastdata = pastdata.to(device=self._device); futuredata = futuredata.to(device=self._device)
        
        self._optimizer.zero_grad()
        if self._network_type == 'SRVNet':
            _pastproject = self._lobe(pastdata); _futureproject = self._lobe(futuredata)
        else:
            _pastproject = self._lobe(pastdata); _futureproject = self._lobe_timelagged(futuredata)
            
        _loss = loss_function(pastdata=_pastproject, futuredata=_futureproject, criterion=self._network_type, 
                              epsilon=self.__epsilon, method=self.__covariance_method)
        _loss.backward()
        self._optimizer.step()
        self._train_scores.append((self.__step, (-_loss).item()))

        if self._network_type == 'SRVNet':
            rev_propogator = reverse_propogator(pastdata=_pastproject, futuredata=_futureproject, epsilon=self.__epsilon, 
                                                method=self.__covariance_method)
            _eigenval, _ = torch.linalg.eigh(rev_propogator)
            self._srvnet_train_eigenvals.append((self.__step, torch.flip(_eigenval[-int(_pastproject.shape[1]):], dims=[0]).detach().numpy()))
        self.__step +=1

        return self
    

    def validate(self, validation_past, validation_future):

        """
        Validate the model using validation dataset to get validation score.
        """

        self._lobe.eval()
        if not self._network_type == 'SRVNet':
            self._lobe_timelagged.eval()

        if isinstance(validation_past, np.ndarray):
            validation_past = torch.from_numpy(validation_past.astype(np.float32))
        if isinstance(validation_future, np.ndarray):
            validation_future = torch.from_numpy(validation_future.astype(np.float32))
        validation_past = validation_past.to(device=self._device)
        validation_future = validation_future.to(device=self._device)

        with torch.no_grad():
            if self._network_type == "SRVNet":
                _val_pastproject = self._lobe(validation_past)
                _val_futureproject = self._lobe(validation_future)
            else:
                _val_pastproject = self._lobe(validation_past)
                _val_futureproject = self._lobe_timelagged(validation_future)
            _val_score = loss_function(pastdata=_val_pastproject, futuredata=_val_futureproject, criterion=self._network_type, 
                              epsilon=self.__epsilon, method=self.__covariance_method)
            if self._network_type == 'SRVNet':
                rev_propogator = reverse_propogator(pastdata=_val_pastproject, futuredata=_val_futureproject, 
                                                    epsilon=self.__epsilon, method=self.__covariance_method)
                _eigenval, _ = torch.linalg.eigh(rev_propogator)
                _eigenval = torch.flip(_eigenval[-int(_val_pastproject.shape[1]):], dims=[0]).detach()
                return -_val_score.item(), _eigenval.numpy()
            else:
                return -_val_score.item()
    


    def fit(self, train_loader, num_epochs=1, validation_loader=None):

        self.__step = 0

        for epoch in range(num_epochs):

            for pastdata, futuredata in train_loader:
                self.partial_fit(pastdata=pastdata.to(device=self._device), futuredata=futuredata.to(self._device))
            if self._network_type == "SRVNet":
                print("==>epoch={}, training process={:.2f}%, the training loss function={}, eigenvalues:{};".format(epoch, 100*(epoch+1)/num_epochs, self._train_scores[-1][1], self._srvnet_train_eigenvals[-1][1]))
            else:
                print("==>epoch={}, training process={:.2f}%, the training loss function={};".format(epoch, 100*(epoch+1)/num_epochs, self._train_scores[-1][1]))
            
            if not validation_loader == None:
                validation_score = []; validation_eigenval = []
                for val_pastdata, val_futuredata in validation_loader:
                    if self._network_type == 'SRVNet':
                        score, eigenval = self.validate(validation_past=val_pastdata, validation_future=val_futuredata)
                        validation_score.append(score); validation_eigenval.append(eigenval)
                    else:
                        score = self.validate(validation_past=val_pastdata, validation_future=val_futuredata)
                        validation_score.append(score)
                validation_score = torch.Tensor(validation_score)
                self._validation_score.append((self.__step, torch.mean(validation_score).item()))
                if len(validation_eigenval) != 0:
                    validation_eigenval = np.array(validation_eigenval)
                    validation_eigenval = torch.Tensor(validation_eigenval)
                    self._srvnet_validate_eigenvals.append((self.__step, torch.mean(validation_eigenval, dim=0)))
                print("==>epoch={}, training process={:.2f}%, the validation loss function={};".format(epoch, 100*(epoch+1)/num_epochs, self._validation_score[-1][1]))
        
        if self._network_type == "SRVNet":
            self._normal_mean, self._normal_eigenvecs = self._normalize(dataloader=train_loader)
        return self
    
    

    def fetch_model(self):
        lobe = deepcopy(self._lobe)
        if self._network_type == "SRVNet":
            lobe_lagged = deepcopy(self._lobe)
        else:
            lobe_lagged = deepcopy(self._lobe_timelagged)
        return Network_Model(lobe, lobe_lagged, device=self._device)
    
    

    def transform(self, data, instantaneous=True):
        """
        Project trajectories data to leart collective variables.
        For SRVNet, the normalization of network output will be done before final projection.
        """
        if self._network_type == 'SRVNet':
            net = deepcopy(self.lobe); net.eval()
            project_data = []
            for _data in map_data(data=data, device=self._device):
                project_data.append(net(_data).detach().cpu().numpy())
            output = []
            for i in range(len(project_data)):
                _output = np.dot((project_data[i]-self._normal_mean), self._normal_eigenvecs)
                output.append(_output)
            return output if len(output) > 1 else output[0]
        else:
            if instantaneous:
                net = self.lobe
            else:
                net = self.lobe_timelagged
            net.eval()
            output = []
            for _data in map_data(data=data, device=self._device):
                output.append(net(_data).detach().cpu().numpy())
            return output if len(output) > 1 else output[0]



