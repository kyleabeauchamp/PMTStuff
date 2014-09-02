import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel
import numpy as np
import mdtraj as md
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 50
n_choose = 40
stride = 1
lag_time = 4

trj0, trajectories, filenames = load_trajectories(stride=stride)
t = trajectories[0].join(trajectories[1:])

d = np.array([md.rmsd(s, s, i) for i in range(len(s))])

lam, ev = np.linalg.eigh(d)

lam2 = lam * 1.0
lam2[abs(lam) < sorted(abs(lam))[::-1][100]] = 0.0
d2 = ev.dot(lam2[:, np.newaxis] * ev.T)
np.linalg.norm(d2 - d)
sum(abs(lam2) > 0)
abs(d2 - d).max()
