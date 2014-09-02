import scipy.optimize
import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.ghmm, mixtape.ntica
import numpy as np
import mdtraj as md
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

stride = 1
trj0, trajectories, filenames = load_trajectories(stride=stride)
trj0 = md.load("./system.subset.pdb")
trj = trajectories[0]

rmsd0 = md.rmsd(trj, trj0, 0)
rmsd1 = md.rmsd(trj, trj, 0)

mean(rmsd0[0:-1] * rmsd1[1:])
mean(rmsd1[0:-1] * rmsd0[1:])

X = [np.array([rmsd0, rmsd1]).T]

ntica = mixtape.ntica.NtICA()
#ntica._lambda = lam
#ntica._lambda = np.array([-0.1, 0.1])
ntica.fit(X)
ntica.offset_correlation_

X0, X1 = ntica.X0, ntica.X1

lam = np.zeros(2)
Q0 = np.exp(-X0.dot(lam))
Q1 = np.exp(-X1.dot(lam))

mu = (X0.T.dot(Q0) + X1.T.dot(Q1)) / (Q0.sum() + Q1.sum())

delta0 = X0 - mu
delta1 = X1 - mu

sigma = (delta0 * Q0[:, np.newaxis]).T.dot(delta0) + (delta1 * Q1[:, np.newaxis]).T.dot(delta1)
sigma /= (Q0.sum() + Q1.sum())

tcorr0 = (delta0 * Q0[:, np.newaxis]).T.dot(delta1)
tcorr0 /= (Q0.sum())


tcorr1 = (delta1 * Q1[:, np.newaxis]).T.dot(delta0)
tcorr1 /= (Q1.sum())


def eqns(lam):
    pad = lambda x: np.pad(x, (1, 0), mode='constant')  # Helper function inserts zero before first element
    lam = pad(lam)
    ind = np.tril_indices(len(lam), -1)
    Q0 = np.exp(-X0.dot(lam))
    Q1 = np.exp(-X1.dot(lam))

    mu = (X0.T.dot(Q0) + X1.T.dot(Q1)) / (Q0.sum() + Q1.sum())

    delta0 = X0 - mu
    delta1 = X1 - mu

    sigma = (delta0 * Q0[:, np.newaxis]).T.dot(delta0) + (delta1 * Q1[:, np.newaxis]).T.dot(delta1)
    sigma /= (Q0.sum() + Q1.sum())

    tcorr0 = (delta0 * Q0[:, np.newaxis]).T.dot(delta1)
    tcorr0 /= (Q0.sum())


    tcorr1 = (delta1 * Q1[:, np.newaxis]).T.dot(delta0)
    tcorr1 /= (Q1.sum())

    return (tcorr0 - tcorr1)[ind]
    

results = scipy.optimize.root(eqns, zeros(1), method='krylov', tol=1E-16)


msm = mixtape.markovstatemodel.MarkovStateModel()
assignments = [np.array([0, 0, 0, 1, 1, 1, 1, 1, 0]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1])]
msm.fit(assignments)



a = msm.populations_ * msm.transmat_.T
b = msm.populations_[:, np.newaxis] * msm.transmat_
a - b

c = msm.countsmat_.sum(1)[:, np.newaxis] * msm.transmat_
d = msm.countsmat_.sum(1) * msm.transmat_.T
c - d

C = msm.countsmat_
T = msm.transmat_
N0 = C.sum(0)
N1 = C.sum(1)
p = msm.populations_

rho0 = p / N0

N0 * T
N0[:, None] * T
N0 * T.T
N0[:, None] * T.T
N1 * T
N1[:, None] * T
N1 * T.T
N1[:, None] * T.T
