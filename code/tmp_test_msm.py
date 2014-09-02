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

tica = mixtape.tica.tICA()
tica.fit(X)


X = np.array([rmsd0, rmsd1]).T
lam = np.zeros(2)
pops = np.exp(-X.dot(lam))
c0 = rmsd0[0:-1].dot(pops[1:] * rmsd1[1:]) / pops[1:].sum()
c1 = rmsd1[0:-1].dot(pops[1:] * rmsd0[1:]) / pops[1:].sum()

def eqn(lam):
    pops = np.exp(-X.dot(lam))
    c0 = rmsd0[0:-1].dot(pops[1:] * rmsd1[1:]) / pops[1:].sum()
    c1 = rmsd1[0:-1].dot(pops[1:] * rmsd0[1:]) / pops[1:].sum()    
    return np.array([c0 - c1, 0.0])

results = scipy.optimize.root(eqn, np.zeros(2), method='krylov', tol=1E-16)
lam = results["x"]
