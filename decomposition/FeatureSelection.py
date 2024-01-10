# Author: Yunrui Qiu <yunruiqiu@gmail.com>
# Contributors: Yue Wu <wu678@wisc.edu>
# The implementation of rao_blackwell_ledoit_wolf algorithm is inspired by msmbuilder 2022
# Copyright (c) 2023, Unveristy of Wisconsin-Madison
# All rights reserved.

import numpy as np
from tqdm import trange, tqdm
import warnings
import os
import pyemma.coordinates
from pyemma.coordinates.transform import nystroem_tica
from pyemma.coordinates.transform.nystroem_tica import oASIS_Nystroem


class correlation_matrix(object):

    def __init__(self, lagtime=100, reverse=True, shrinkage=None):
        self.__num_sequence = 0
        self.__num_datapoints = 0
        self.__num_features = None
        self.__reverse = reverse
        self.__shrinkage = shrinkage
        self._lagtime = lagtime
        self._c00 = None; self._c0t = None; self._ctt = None
        self._mean0 = None; self._meant = None; self._mean_all = None
        self._covariance = None
        self._timelagged_covariance = True

    
    @property
    def covariance_matrix(self):
        return self._covariance
    
    @property
    def timelagged_covariance(self):
        return self._timelagged_covariance
    
    @property
    def c00(self):
        return self._c00
    
    @property
    def c0t(self):
        return self._c0t
    
    @property
    def ctt(self):
        return self._ctt
    

    def _fit(self, data):

        self.__num_sequence += 1
        self.__num_datapoints += data.shape[0]
        if self.__num_features == None:
            self.__num_features = data.shape[1]
        else:
            assert data.shape[1] == self.__num_features, "The number of features for the sequences is not consistent: {} != {}.".format(data.shape[1], self.__num_features)
        
        if data.shape[0] <= self._lagtime:
            warnings.warn("The length of the data is shorter than the lagtime: {} < {}.".format(data.shape[0], self._lagtime))
        
        if self._c00 is None and self._c0t is None and self._ctt is None:
            self._c00 = np.zeros((self.__num_features, self.__num_features))
            self._c0t = np.zeros((self.__num_features, self.__num_features))
            self._ctt = np.zeros((self.__num_features, self.__num_features))
            self._mean0 = np.zeros(self.__num_features)
            self._meant = np.zeros(self.__num_features)
            self._mean_all = np.zeros(self.__num_features)

        self._c00 += np.dot(data[:-self._lagtime].T, data[:-self._lagtime])
        self._ctt += np.dot(data[self._lagtime:].T, data[self._lagtime:])
        self._c0t += np.dot(data[:-self._lagtime].T, data[self._lagtime:])
        self._mean0 += np.sum(data[:-self._lagtime], axis=0)
        self._meant += np.sum(data[self._lagtime:], axis=0)
        self._mean_all += np.sum(data, axis=0)
    

    @staticmethod
    def rao_blackwell_ledoit_wolf(covmatrix, num_data):
        """
        Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance matrix.
        [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
        estimation of high dimensional covariance matrices" ICASSP (2009)
        This part is inspired by MSMBuilder 2022;
        """
        matrix_dim = covmatrix.shape[0]
        assert covmatrix.shape == (matrix_dim, matrix_dim), "The input covariance matrix does not have the squared shape;"

        alpha = (num_data-2)/(num_data*(num_data+2))
        beta = ((matrix_dim+1)*num_data - 2) / (num_data*(num_data+2))

        trace_covmatrix_squared = np.sum(covmatrix*covmatrix) 
        U = ((matrix_dim * trace_covmatrix_squared / np.trace(covmatrix)**2) - 1)
        rho = min(alpha + beta/U, 1)

        F = (np.trace(covmatrix) / matrix_dim) * np.eye(matrix_dim)
        return (1-rho)*covmatrix + rho*F, rho


    def fit(self, traj):

        for i in tqdm(range(len(traj)), desc="Fit correlation matrix"):
            self._fit(traj[i])
        average_num = (self.__num_datapoints - self._lagtime * self.__num_sequence)
        self._mean0 /= float(average_num); self._meant /= float(average_num)
        self._mean_all /= self.__num_datapoints
        if self.__reverse:

            self._covariance =  (self._c00 + self._ctt) / 2 / float(average_num) - np.outer((self._mean0+self._meant)/2, (self._mean0+self._meant)/2)
            self._timelagged_covariance = (self._c0t + self._c0t.T) / 2 / float(average_num) - np.outer((self._mean0+self._meant)/2, (self._mean0+self._meant)/2)

            self._c00 = (self._c00) / float(average_num) - np.outer(self._mean0, self._mean0)
            self._c0t = (self._c0t + self._c0t.T) / 2 / float(average_num) - np.outer(self._mean_all, self._mean_all)
            self._ctt = (self._ctt) /float(average_num) - np.outer(self._meant, self._meant)

        else:
            self._c00 = (self._c00) / float(average_num) - np.outer(self._mean0, self._mean0)
            self._c0t = (self._c0t) / float(average_num) - np.outer(self._mean0, self._meant)
            self._ctt = (self._ctt) / float(average_num) - np.outer(self._meant, self._meant)

            self._covariance = (self._c00 + self._ctt) / 2 
            self._timelagged_covariance = self._c0t.copy()


        if self.__shrinkage is None:
            self._covariance, self.shrinkage = self.rao_blackwell_ledoit_wolf(self._covariance, self.__num_datapoints)
        else:
            self.shrinkage = self.__shrinkage
            F = (np.trace(self._covariance)/self.__num_features) * np.eye(self.__num_features)
            self._covariance = (1-self.shrinkage)*self._covariance + self.shrinkage*F



class spectral_oasis(object):
    """Select feature by spectral-oasis algorithm
    
    Implement nystrom algorithm with modified targeted function to approximate 
    covariance matrix / reversed time-lagged covarianace matrix. Select features
    to achieve the optimal and accurate reconstructions.

    Parameters
    ----------
    num_select: int, default=100
        Number of features to select.
    num_every_iter: int, default=5
        Number of features chosen by spectral-oasis within one iteration.
    method: string, default='spectral-oasis'
        Method used for feature selection, choice: 'random', 'oasis', 'spectral-oasis'.
    matrix: numpy.array, default=None
        The matrix to implement feature selection algorithms, should keep shape: (num_features, num_features),
        if not given, will automatically adopt the reversed self-covaraince matrix.
    covariance: bool, default=True
        If True, reversed covariance matrix will be used to perform feature selection algorithm;
        If False, reversed time-lagged covariance matrix will be used. Need to set lagtime parameters in select method.
    shrinkage: float, default=None
        The covariance shrinkage intensity (range 0-1). If shrinkage is not specified (the default) it is estimated 
        using an analytic formula (the Rao-Blackwellized Ledoit-Wolf estimator).
    random_seed: int, default=42
        The random seed to decide the initial features.

    Attributes
    ----------
    select_columns: list-like, shape: (num_select,)
        index list for the selected columns/features.
    error: list-like, shape: (int(num_select/num_every_iter), )
        error value when including different features.
    matrix: array-like, shape: (num_features, num_features)
        the matrix used to perform feature selection algorithm
    
        
    Methods
    ----------
    select(trajs=None, lagtime=None): method to perform feature selection
    parameter trajs: the trajectories needed to do feature slection, list-like, shape: (num_sequences, )(num_datapoints, num_features).
    parameter lagtime: lagtime to use reversed time-lagged covariance matrix and perform the corresponding selections. 


    References
    ----------
    ..[1]. Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. ICASSP (2009)
    ..[2]. Clementi. C et. al. J. Chem Theory Comput. 2018 14 (5), 2771-2783
    ..[3].  Schwantes, Christian R., and Vijay S. Pande. J. Chem Theory Comput. 9.4 (2013): 2000-2009.
    
    """

    def __init__(self, num_select=100, num_every_iter=5, method='spectral-oasis', matrix=None, covariance=True, shrinkage=None, random_seed=42):
        
        self._matrix = matrix
        self._num_select = num_select
        self._num_every_iter = num_every_iter
        self._method = method
        self._error = None
        self._select_columns = None
        self._covariance = covariance
        self.__shrinkage = shrinkage
        self.__random_seed = random_seed

    
    @property
    def select_columns(self):
        return self._select_columns
    
    @property
    def error(self):
        return self._error
    
    @property
    def matrix(self):
        return self._matrix

    
    def __test_trajs_matrix(self, trajs, lagtime=None):
        
        if trajs is None and self._matrix is None:
            raise ValueError("At least one of the matrix or trajectories need to be the input to perform spectral-oasis;")
        
        ## Input the matrix to do nystrom algorithm
        if not self._matrix is None:
            if trajs is None:
                diff = 0
            else:
                if lagtime is None and self._covariance:
                    matrix = correlation_matrix(lagtime=100, reverse=True, shrinkage=self.__shrinkage)
                    matrix.fit(trajs)
                    diff = np.sum((matrix.covariance_matrix - self._matrix)**2)
                else:
                    matrix = correlation_matrix(lagtime=lagtime, reverse=True, shrinkage=self.__shrinkage)
                    matrix.fit(trajs)
                    diff = np.sum((matrix.timelagged_covariance - self._matrix)**2)
            if diff > 1e-3:
                raise ValueError("The input matrix and input trajectories are not consistent;")
        
        ## Did not input the matrix
        else:
            if self._covariance and lagtime is None:
                matrix = correlation_matrix(lagtime=100, reverse=True, shrinkage=None)
                matrix.fit(trajs)
                self._matrix = matrix.covariance_matrix
            elif self._covariance and lagtime is not None:
                raise ValueError("The argument covariance has conflict with argument lagtime;")
            elif not self._covariance and lagtime is None:
                raise ValueError("The argument covariance has conflict with argument lagtime;")
            else:
                matrix = correlation_matrix(lagtime=lagtime, reverse=True, shrinkage=None)
                matrix.fit(trajs)
                self._matrix = matrix.timelagged_covariance

        return self


    def select(self, trajs=None, lagtime=None):

        self.__test_trajs_matrix(trajs, lagtime)
        diag = np.diag(self._matrix)
        dim = self._matrix.shape[0]
        np.random.seed(self.__random_seed); initial_cols = np.random.choice(dim, self._num_every_iter, replace=False)
        self._select_columns = initial_cols
        c0_k = self._matrix[:, initial_cols]
        oasis = oASIS_Nystroem(d=diag, C_k=c0_k, columns=initial_cols)
        oasis.set_selection_strategy(strategy=self._method, nsel=self._num_every_iter, neig=4)
        self._error = [np.sum(np.abs(oasis.error))]

        for i in tqdm(range(int(self._num_select/self._num_every_iter)-1), desc="feature selection"):
            newcol = oasis.select_columns()
            c = self._matrix[:, newcol]
            oasis.add_columns(C_k_new=c, columns_new=newcol)
            self._select_columns = np.append(self._select_columns, newcol)
            self._error.append(np.sum(np.abs(oasis.error)))
        
        return self




