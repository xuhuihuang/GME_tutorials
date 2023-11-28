# Author: Siqin Cao <siqincao@gmail.com>
# Copyright (c) 2023, University of Wisconsin-Madison and the authors
# All rights reserved.

""" Integrative Generalized Master Equation (IGME)
Use the Least Square fitting algorithm to build IGME model(s) 

The theory of IGME (http://doi.org/10.1063/5.0167287):
    T(t) = A \hat{T}^t

The implementation in this python script:
  IGME._theory == 0 (default):
    ln T(t) = ln A + t ln \hat{T} 
  IGME._theory == 2:
    constructed with    ln T(t) = ln A + t ln \hat{T} 
    RMSE computed with  T(t) = A \hat{T}^t
  IGME._theory == 3:
    constructed with    ln T(t) = ln A + t ln \hat{T}
    RMSE computed with  T(t) = A^0.5 \hat{T}^t A^0.5
"""

import numpy
import sys

class IGMENS :
    @staticmethod
    def matrix_log_accurate(m) :
        ev, vr = numpy.linalg.eig(m)
        return numpy.matmul(vr, numpy.matmul(numpy.diag(numpy.log(ev)), numpy.linalg.inv(vr)))

    @staticmethod
    def matrix_log_approx(m, order=10) :
        res = m - numpy.identity(len(m))
        ret = 0
        temp = res
        for i in range(1, order+1) :
            ret += temp / i;
            temp = numpy.matmul(temp, -res)
        return ret
        
    @staticmethod
    def matrix_log(m, logarithm_approx_order=0) :
        if (logarithm_approx_order>=2) & (logarithm_approx_order<=100) :
            return IGMENS.matrix_log_approx(m, logarithm_approx_order)
        else :
            return IGMENS.matrix_log_accurate(m)

    @staticmethod
    def matrics_log(ms, logarithm_approx_order=0) :
        ret = []
        for i in range(len(ms)) :
            ret.append(IGMENS.matrix_log(ms[i], logarithm_approx_order))
        return ret

    @staticmethod
    def matrix_exp(m) :
        ev, vr = numpy.linalg.eig(m)
        return numpy.matmul(vr, numpy.matmul(numpy.diag(numpy.exp(ev)), numpy.linalg.inv(vr)))

    @staticmethod
    def matrics_exp(ms) :
        ret = []
        for i in range(len(ms)) :
            ret.append(IGMENS.matrix_exp(ms[i]))
        return ret

    @staticmethod
    def lsfs_xy(x, y, dimension) :
        length = len(x)
        if (length>len(y[0])) :
            length = len(y[0])
        M = numpy.zeros([2*dimension+1,2*dimension+1])
        Z = numpy.zeros([2*dimension+1,1])
        R = numpy.zeros([dimension])
        sx = numpy.mean(x)
        sxx = numpy.dot(x, x) / length
        for i in range(dimension) :
            M[i,i] = sx
            M[i+dimension,i] = 1
            M[i+dimension,i+dimension] = sx
            M[i,i+dimension] = sxx
            M[i,2*dimension] = 1
            M[2*dimension,i+dimension] = 1
            Z[i] = numpy.dot(x, y[i]) / length
            Z[i+dimension] = numpy.mean(y[i])
            sy = numpy.mean(y[i])
            syy = numpy.dot(y[i], y[i]) / length
            sxy = Z[i]
            R[i] = (sxy - sx*sy) / numpy.sqrt((sxx-sx*sx) * (syy - sy*sy))
        M[2*dimension,2*dimension] = 0
        Z[2*dimension] = 0
        return numpy.matmul(numpy.linalg.inv(M), Z), R

    @staticmethod
    def matrics_lsfs(mat, dimension, begin, end) :
        A = numpy.zeros([dimension,dimension])
        B = numpy.zeros([dimension,dimension])
        R = numpy.zeros([dimension,dimension])
        x = numpy.zeros([end-begin+1])
        for i in range(dimension) :
            y = numpy.zeros([dimension, end-begin+1])
            for k in range(begin, end+1) :
                x[k-begin] = k+1
                for j in range(dimension) :
                    y[j][k-begin] = mat[k][i][j]
            lsfs_xy_ret, lsfs_xy_r = IGMENS.lsfs_xy(x, y, dimension)
            for l in range(dimension) :
                A[i][l] = lsfs_xy_ret[l]
                B[i][l] = lsfs_xy_ret[l+dimension]
                R[i][l] = lsfs_xy_r[l]
        return A, B, R

    @staticmethod
    def print_matrix_in_csv(m, end="\n") :
        for i in range(len(m)) :
            for j in range(len(m[i])) :
                if (i>0) | (j>0) :
                    print(",", end="")
                print(m[i][j], end="")
        print(end, end="")


class IGME(object) :

    """ Integrative Generalized Master Equation

    Parameters
    ----------
    logarithm_approx_order : int, default: 0
        When this parameter is set >=2 and <=100, the logarithm will be
        computed with Taylor expansions up to the order of this parameter.
        When this parameter is below 2 or above 100, the approximation will be
        disabled and the accurate logarithm will be used.
        This parameter is used to compute matrix logarithm when eigenvalues
        contain negative components. In such cases, logarithm of matrix will
        contain complex elements.

    Attributes
    ----------
    lnTh : array like, shape = (n_state, n_state)
        The logorithm of \hat{T} matrix
    Th : array like, shape = (n_state, n_state)
        The \hat{T} matrix
    lnA : array like, shape = (n_state, n_state)
        The logorithm of A matrix
    A : array like, shape = (n_state, n_state)
        The A matrix
    R : array like, shape = (n_state, n_state)
        A matrix of Pearson Correlation Coefficients
    rmse : real
        The RMSE between input and IGME predictions
    timescales : array like, shape = (n_state - 1)
        The implied timescales of \hat{T} matrix. This is also the long-time
        limit of implied timescales of IGME
    M0 : array like, shape = (2, n_state, n_state)
        The time integral of memory kernels.
        M0[0] = \dot{T}(0) - ln \hat{T} 
        M0[1] = T(1)^-1 \dot{T}(1) - ln \hat{T} 
    mik : array like, shape = (2)
        The root square mean of all elements of M0
        mik[0] is computed from M0[0], mik[1] is computed from M0[1]

    Reference
    ----------

    """

    def __init__(self, logarithm_approx_order=0) :
        self._logarithm_approx_order = logarithm_approx_order
        self._theory = 0
        self._dim   = 0
        self._begin = 0
        self._end   = 0
        self._lnTh  = 0
        self._Th    = 0
        self._lnA   = 0
        self._A     = 0
        self._R     = 0
        self._M0    = 0
        self._mik   = 0
        self._timescales = 0
        self._rmse  = 0

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end

    @property
    def lnTh(self):
        return self._lnTh

    @property
    def A(self):
        return self._A

    @property
    def lnA(self):
        return self._lnA

    @property
    def Th(self):
        return self._Th

    @property
    def R(self):
        return self._R

    @property
    def rmse(self):
        return self._rmse

    @property
    def timescales(self):
        return self._timescales

    @property
    def M0(self):
        return self._M0

    @property
    def mik(self):
        return self._mik

    def fit(self, input_data, begin, end, rmse_weighted_by_sp=True) :
        """ Construct an IGME model based on the input data

        Parameters
        ----------
        input_data : array like
            shape = (n_lagtime, n_state^2) or (n_lagtime, n_state, n_state)
            Time dependent TPMs from \delta t to n_lagtime \delta t 
            i.e., input_data[0] = trajectory_TPM(\delta t),
                  input_data[1] = trajectory_TPM(2 \delta t),
                  input_data[n_lagtime-1] = trajectory_TPM(n_lagtime \delta t),
        begin : int
            The begin time used to fit IGME. Begin with 1. Should exceed
            the memory relaxation time.
        end : int
            The end time used to fit IGME. Begin with 1. Length of input_data
            will be used instead if end is smaller than begin
        rmse_weighted_by_sp : boolean, default: True
            Allow RMSE weighted by stationary populations
            The stationary populations are computed by sqrt(T[begin]*T[end])^(1e9)

        Returns
        -------
        self : object
            returns the instance itself

        """
        if (numpy.ndim(input_data)==2) :
            self._dim = int(numpy.sqrt(len(input_data[0])))
        elif (numpy.ndim(input_data)==3) :
            if (len(input_data[0]) != len(input_data[0][0])) :
                self._dim = int(numpy.sqrt(len(input_data[0]) * len(input_data[0][0])))
            else :
                self._dim = len(input_data[0])
        else :
            self._dim = 0
        reshaped_input_data = input_data.reshape([len(input_data), self._dim, self._dim]) 
        self._begin = begin
        self._end = end
        if (self._dim <= 0) :
            return 0
        lnTPMs = IGMENS.matrics_log(reshaped_input_data, self._logarithm_approx_order)
        self._lnA, self._lnTh, self._R = IGMENS.matrics_lsfs(lnTPMs, self._dim, self._begin-1, self._end-1)
        self._A = IGMENS.matrix_exp(self._lnA)
        self._Th = IGMENS.matrix_exp(self._lnTh)
        ev = -numpy.real(numpy.linalg.eigvals(self._lnTh))
        ev.sort()
        self._timescales = (1 / ev[1:])
        self._rmse = self._compute_rmse(reshaped_input_data, rmse_weighted_by_sp)
        self._M0 = [ 0, 0 ]
        self._M0[0] = IGMENS.matrix_log(reshaped_input_data[0], self._logarithm_approx_order) - self._lnTh
        self._M0[1] = numpy.matmul(numpy.linalg.inv((reshaped_input_data[0] + reshaped_input_data[1])*0.5), reshaped_input_data[1] - reshaped_input_data[0]) - self._lnTh
        self._mik = [ 0, 0 ]
        self._mik[0] = numpy.linalg.norm(self._M0[0]) / self._dim
        self._mik[1] = numpy.linalg.norm(self._M0[1]) / self._dim
        return self

    def predict(self, begin, end) :
        """ Generate IGME predictions of TPMs at given range of lagtime

        Parameters
        ----------
        begin : int
            The begin time used to fit IGME. Begin with 1. Should exceed
            the memory relaxation time.
        end : int
            The end time used to fit IGME. Begin with 1. Length of input_data
            will be used instead if end is smaller than begin

        Returns
        -------
        predicted_TPMs : array like, shape = (end-begin+1, n_state, n_state)
            The predicted TPMs at given range of lagtime

        """
        if (self._dim <= 0) :
            return 0
        ret = [];
        for i in range(begin-1, end) :
            if (self._theory == 2) :
                ret.append(numpy.matmul(self._A, IGMENS.matrix_exp(numpy.multiply(self._lnTh, i))))
            elif (self._theory == 3) :
                Asqrt = IGMENS.matrix_exp(numpy.multiply(self._lnA, 0.5))
                ret.append(numpy.matmul(Asqrt, numpy.matmul(IGMENS.matrix_exp(numpy.multiply(self._lnTh, i)), Asqrt)))
            else :
                ret.append(IGMENS.matrix_exp(numpy.multiply(self._lnTh, i) + self._lnA))
        return ret

    def _compute_rmse(self, data, rmse_weighted_by_sp=True) :
        if (self._dim <= 0) :
            return 0
        diff = self.predict(1, len(data)) - data
        if (rmse_weighted_by_sp) :
            sp_matrix = numpy.diag(IGMENS.matrix_exp(numpy.multiply(IGMENS.matrix_log(data[int((self._begin+self._end)/2)], self._logarithm_approx_order), 1e9).reshape([self._dim, self._dim]))[0]);
            for i in range(len(diff)) :
                diff[i] = numpy.matmul(sp_matrix, diff[i])
        return numpy.sqrt(numpy.square(numpy.mean(diff)) + numpy.square(numpy.std(diff)))

    def _init_output_dic(self) :
        dic = {}
        dic['begin']    = []
        dic['end']      = []
        dic['lnTh']     = []
        dic['Th']       = []
        dic['lnA']      = []
        dic['A']        = []
        dic['R']        = []
        dic['rmse']     = []
        dic['timescales'] = []
        dic['M0']       = []
        dic['mik']      = []
        return dic

    def output(self, dic=0) :
        """ Output the IGME model to a dictionary

        Parameters
        ----------
        dic : dictionary, optional, default: 0
            The dictionary to append current IGME model
            Ignore this parameter if you want to generate a new dictionary

        Returns
        -------
        dic : dictionary
            A dictionary include all the fitted results of IGME
            If input parameter "dic" is given, then output will be the same as
            the input parameter

        """
        if (dic == 0):
            dic = self._init_output_dic()
        dic['begin'].append(self._begin)
        dic['end'].append(self._end)
        dic['lnTh'].append(self._lnTh)
        dic['Th'].append(self._Th)
        dic['lnA'].append(self._lnA)
        dic['A'].append(self._A)
        dic['R'].append(self._R)
        dic['rmse'].append(self._rmse)
        dic['timescales'].append(self._timescales)
        dic['M0'].append(self._M0)
        dic['mik'].append(self._mik)
        return dic

    def _from_output(self, dic, index=0) :
        """ copy IGME model from output dictionary: dic[index]

        Parameters:
        ----------
        dic : dictionary
            The dictionary of output(s) generated with IGME.output()
        index : int
            The element in dic to copy 

        Returns
        -------
        self : object
            returns the instance itself
        """
        self._dim        = len(dic['Th'][index])
        self._begin      = dic['begin'][index]
        self._end        = dic['end'][index]
        self._lnTh       = dic['lnTh'][index]
        self._Th         = dic['Th'][index]
        self._lnA        = dic['lnA'][index]
        self._A          = dic['A'][index]
        self._R          = dic['R'][index]
        self._M0         = dic['M0'][index]
        self._mik        = dic['mik'][index]
        self._timescales = dic['timescales'][index]
        self._rmse       = dic['rmse'][index]
        return dic

    def fit_output(self, input_data, begin, end, rmse_weighted_by_sp=True) :
        """ Construct an IGME model based on the input data

        Parameters
        ----------
        input_data : array like
            shape = (n_lagtime, n_state^2) or (n_lagtime, n_state, n_state)
            Time dependent TPMs from \delta t to n_lagtime \delta t 
            i.e., input_data[0] = trajectory_TPM(\delta t),
                  input_data[1] = trajectory_TPM(2 \delta t),
                  input_data[n_lagtime-1] = trajectory_TPM(n_lagtime \delta t),
        begin : int
            The begin time used to fit IGME. Begin with 1. Should exceed
            the memory relaxation time.
        end : int
            The end time used to fit IGME. Begin with 1. Length of input_data
            will be used instead if end is smaller than begin
        rmse_weighted_by_sp : boolean, default: True
            Allow RMSE weighted by stationary populations
            The stationary populations are computed by sqrt(T[begin]*T[end])^(1e9)

        Returns
        -------
        dic : dictionary
            A dictionary include all the fitted results of IGME

        """
        return self.fit(input_data, begin, end, rmse_weighted_by_sp).output()

    def _print_number_array(self, a, end=",", file=sys.stdout) :
        print("'", end="", file=file)
        for i in range(len(a)) :
            print(a[i], end = " " if i<len(a)-1 else "", file=file)
        print("'", end=end, file=file)

    def print_output(self, out, begin=0, end=0, comma=",", file=sys.stdout, _print_title=True) :
        if _print_title :
            print("#begin,end,rmse,timescales,lnTh,lnA,mik", file=file)
        if (end<=begin) | (end>len(out['begin'])-1) :
            end = len(out['begin']) - 1
        for i in range(begin, end+1) :
            dim = len(out['lnTh'][i])
            print(out['begin'][i], end=comma, file=file)
            print(out['end'][i], end=comma, file=file)
            print(out['rmse'][i], end=comma, file=file)
            self._print_number_array(out['timescales'][i], end=comma, file=file)
            self._print_number_array(out['lnTh'][i].reshape([dim*dim]), end=comma, file=file)
            self._print_number_array(out['lnA'][i].reshape([dim*dim]), end=comma, file=file)
            print("'"+str(out['mik'][i][0])+" "+str(out['mik'][i][1])+"'", end='\n', file=file)

    def scan(self, input_data, begin, end, stride=1, rmse_weighted_by_sp=True, debug=False) :
        """ Scan IGME models within the given range of lagtime

        Parameters
        ----------
        input_data : array like
            shape = (n_lagtime, n_state^2) or (n_lagtime, n_state, n_state)
            Time dependent TPMs from \delta t to n_lagtime \delta t 
            i.e., input_data[0] = trajectory_TPM(\delta t),
                  input_data[1] = trajectory_TPM(2 \delta t),
                  input_data[n_lagtime-1] = trajectory_TPM(n_lagtime \delta t),
        begin : int
            The begin time used to scan IGME models. Begin with 1. Should exceed
            the memory relaxation time.
        end : int
            The end time used to scan IGME models. Begin with 1. Length of
            input_data will be used instead if end is smaller than begin
        stride : int, default: 1
            The stride steps int the scanning.
        rmse_weighted_by_sp : boolean, default: True
            Allow RMSE weighted by stationary populations
            The stationary populations are computed by sqrt(T[begin]*T[end])^(1e9)
        debug : boolean, default: False
            If turned on, then all scanned IGME models will be displayed on the
            screen immediately (in CSV format)

        Returns
        -------
        output : dictionary
            returns a dictionary containing all IGME models

        """
        output = self._init_output_dic()
        _print_title = True
        for iend in range(begin+stride, end+1, stride) :
            for ibegin in range(begin, iend, stride) :
                output_itself = self.fit(input_data, ibegin, iend, rmse_weighted_by_sp).output(output)
                if debug :
                    self.print_output(output, begin=len(output['begin'])-1, _print_title=_print_title)
                    _print_title = False
        return output

    def top_outputs(self, dic, n=1, min_its=0, max_its=1e18) :
        """ Extract the top models of an output dictionary.
        Top models are defined by smallest RMSEs.

        Parameters
        ----------
        dic: dictionary
            The full dictionary
        n: int or float
            The top models to output
            n>=1 : the number of models
            n<1 : the percentage of original data

        Return
        ------
        output: dictionary
            returns a dictory of top elements of input dictionary

        """
        total_rec = len(dic['Th'])
        n_rec = n
        if n < 1 :
            n_rec = numpy.array(numpy.ceil(total_rec * n)).astype(numpy.int)
        if n_rec < 1 :
            n_rec = 1
        if n_rec >= total_rec :
            return dic
        ind_rmse = numpy.argsort(dic['rmse'])
        ret = self._init_output_dic()
        temp = IGME()
        n_out = 0
        #for i in range(n_rec) :
        for i in range(total_rec) :
            if numpy.min(numpy.real(dic['timescales'][ind_rmse[i]])) < min_its:
                #print("data ignored: "+str(dic['timescales'][ind_rmse[i]]))
                continue
            elif numpy.max(numpy.real(dic['timescales'][ind_rmse[i]])) > max_its:
                #print("data ignored: "+str(dic['timescales'][ind_rmse[i]]))
                continue
            else :
                temp._from_output(dic, ind_rmse[i])
                temp.output(ret)
                n_out += 1
            if n_out > n_rec:
                break
        return ret

    def _median_outputs(self, dic) :
        """Find the median model, closest to the average ln\hat{T}

        Parameters
        ----------
        dic: dictionary
            The full dictionary

        Return
        ------
        output: dictionary
            returns a dictory of top elements of input dictionary
        """
        Th_average = numpy.mean(dic['lnTh'], axis=0)
        rmse_list = [0] * len(dic['lnTh'])
        for i in range(len(dic['lnTh'])) :
            rmse_list[i] = numpy.linalg.norm(dic['lnTh'][i] - Th_average)
        ind_rmse = numpy.argsort(rmse_list)
        ret = IGME()
        ret._from_output(dic, ind_rmse[0])
        return ret

    def top_model(self, dic, n=1, min_its=0, max_its=1e18) :
        """Find the median model of top models
        Top models have the smallest RMSEs
        The median model has \ln\hat{T} closest to average

        Parameters
        ----------
        dic: dictionary
            The full dictionary
        n: int or float
            The top models to output
            n>=1 : the number of models
            n<1 : the percentage of original data

        Return
        ------
        output: object 
            returns the best IGME object
        """
        result = self.top_outputs(dic, n, min_its, max_its)
        ret = self._median_outputs(result)
        return ret

