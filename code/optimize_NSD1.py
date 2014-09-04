import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel
import numpy as np
import mdtraj as md
from mixtape import ghmm, feature_selection, subset_featurizer
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 50

n_choose = 100
stride = 1
lag_time = 1

trj0, trajectories, filenames = load_trajectories(stride=stride)

train = trajectories[0::2]
test = trajectories[1::2]

#featurizer = mixtape.subset_featurizer.guess_featurizers(trajectories[0][0], n_choose)
featurizer = sklearn.externals.joblib.load("./featurizer-%d.job" % n_choose)

model = mixtape.tica.tICA(lag_time=lag_time, n_components=4)

tica_optimizer = mixtape.feature_selection.Optimizer(featurizer, model)
featurizer = tica_optimizer.optimize(n_iter, train)

#sklearn.externals.joblib.dump(tica_optimizer.featurizer, "./featurizer-%d.job" % n_choose, compress=True)
