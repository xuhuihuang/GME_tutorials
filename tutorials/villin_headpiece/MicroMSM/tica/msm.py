import numpy as np
import time
import multiprocessing
from joblib import Parallel, delayed
from msmbuilder.decomposition import tICA
from msmbuilder.msm import MarkovStateModel
from msmbuilder.cluster import KMeans, MiniBatchKMeans, KCenters

oasis_trajs = np.load('oasis_trajs.npy', allow_pickle=True)

tica = tICA(n_components=4, lag_time=200, kinetic_mapping=True)
tica_trajs = tica.fit_transform(oasis_trajs)
tica_trajs = np.array(tica_trajs, dtype=object)
np.save('tica_trajs.npy', tica_trajs)

cluster = KCenters(n_clusters=200, random_state=42)
clustered_trajs = cluster.fit_transform(tica_trajs)
clustered_trajs = np.array(clustered_trajs, dtype=object)
np.save('clustered_trajs.npy', clustered_trajs)

def find_max_indices(lst, num):
    max_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:num]
    return max_indices

def bootstrapMSM(trajs, lagtime, n_timescales, n_RP, RP_idx, n_states, num_samples_per_run):
    
    bootstrap_indices = np.random.choice(range(len(trajs)), 
                                         size=num_samples_per_run, replace=True)
    bootstrap_trajs = [trajs[i] for i in bootstrap_indices]
    
    states_set = set(np.concatenate(bootstrap_trajs))
    reference_set = set(np.arange(n_states))
    miss_states_idx = sorted(list(reference_set - states_set))
    
    bootstrap_ITS = np.zeros((len(lagtime), n_timescales))
    bootstrap_RP = np.zeros((len(lagtime), n_RP))
            
    for i in range(len(lagtime)):
        start_time = time.time()
        msm = MarkovStateModel(n_timescales=n_timescales, lag_time=lagtime[i], ergodic_cutoff='off', 
                               reversible_type='transpose', verbose=False)
        msm.fit(bootstrap_trajs)        
        bootstrap_ITS[i] = msm.timescales_
        RP_all = np.diag(msm.transmat_)
        if len(miss_states_idx) != 0:
            for index in miss_states_idx:
                RP_all = np.insert(RP_all, index, 0)
#        print("The length of resident probability is %d"%(len(RP_all)))
        bootstrap_RP[i] = RP_all[RP_idx]
        end_time = time.time()
        print('Run with lagtime %d takes %2f s!'%(lagtime[i], end_time - start_time))
    return bootstrap_ITS, bootstrap_RP

num_runs = 50
num_samples_per_run = 150
# num_processes = int(0.2 * multiprocessing.cpu_count())
lagtime = np.arange(5,1501,5)
tpm_lagtime = [100, 200, 400, 500, 800]
n_timescales = 10
n_RP = 8
bootstrap_ITS = np.memmap("bootstrap_ITS.npy", dtype=np.float64, 
                          mode="w+", shape=(num_runs, len(lagtime), n_timescales))
bootstrap_RP = np.memmap("bootstrap_RP.npy", dtype=np.float64, 
                          mode="w+", shape=(num_runs, len(lagtime), n_RP))

for lt in tpm_lagtime:
    msm = MarkovStateModel(n_timescales=n_timescales, lag_time=lt, ergodic_cutoff='off',
                           reversible_type='transpose', verbose=False)
    msm.fit(clustered_trajs)
    TPM = np.array(msm.transmat_)
    np.save('%d_lt_micro_TPM.npy'%(lt), TPM)  
states_idx = find_max_indices(msm.populations_, n_RP)
np.savetxt('states_idx.txt', states_idx, fmt='%d')

def Process(run):
    start_time = time.time()
    result1, result2 = bootstrapMSM(trajs=clustered_trajs, lagtime=lagtime, 
                                    n_timescales=n_timescales, n_RP=n_RP, 
                                    RP_idx=states_idx, n_states=200, 
                                    num_samples_per_run=num_samples_per_run)
    bootstrap_ITS[run] = result1
    bootstrap_RP[run] = result2
    end_time = time.time()
    print("Run %d takes %2f s."%(run, end_time - start_time))
    print("**************************************************************")
    
Parallel(n_jobs=5)(delayed(Process)(run = i) for i in range(num_runs))    
'''
np.save('bootstrap_ITS.npy', bootstrap_ITS)
np.save('bootstrap_RP.npy', bootstrap_RP)
'''