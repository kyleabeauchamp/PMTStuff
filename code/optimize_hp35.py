import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.datasets, mixtape.subset_featurizer, mixtape.feature_selection
import numpy as np
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils
from parameters import load_trajectories


n_iter = 5000
n_choose = 10
lag_time = 1

trj0, trajectories, filenames = load_trajectories()
train = trajectories

featurizer = mixtape.subset_featurizer.guess_featurizers(trajectories[0], n_choose)
tica_optimizer = mixtape.feature_selection.TICAOptimizer(featurizer, lag_time=lag_time)
tica_optimizer.optimize(n_iter, train)

#sklearn.externals.joblib.dump(tica_optimizer.featurizer, "./featurizer-%d.job" % n_choose, compress=True)
