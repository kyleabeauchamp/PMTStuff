import mdtraj as md
import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.datasets, mixtape.subset_featurizer, mixtape.feature_selection
import numpy as np
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 2500
n_choose = 100
lag_time = 4
n_components = 4

trajectories = [md.load("./Trajectories/trj0.h5")]
train = [trajectories[0][0:60000]]

featurizer = mixtape.subset_featurizer.guess_featurizers(trajectories[0][0], n_choose)
#featurizer = sklearn.externals.joblib.load("./featurizer-%d.job" % n_choose)

model = mixtape.tica.tICA(lag_time=lag_time, n_components=n_components)

tica_optimizer = mixtape.feature_selection.Optimizer(featurizer, model)
featurizer = tica_optimizer.optimize(n_iter, train)

sklearn.externals.joblib.dump(featurizer, "./featurizer-%d.job" % n_choose, compress=True)
