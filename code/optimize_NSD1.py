import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel
import numpy as np
import mdtraj as md
from mixtape import ghmm, selector, subset_featurizer, selector
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 50000

n_choose = 100
stride = 1
lag_time = 4

trj0, trajectories, filenames = load_trajectories(stride=stride)

train = trajectories[0::2]
test = trajectories[1::2]

featurizer = build_full_featurizer(trj0, n_choose)  # Doesn't work right now, too many features need to re-optimize later.
#featurizer = sklearn.externals.joblib.load("./featurizer-%d.job" % n_choose)

tica_optimizer = mixtape.selector.TICAOptimizer(featurizer, train, lag_time=lag_time)
tica_optimizer.optimize(n_iter, train)

#sklearn.externals.joblib.dump(tica_optimizer.featurizer, "./featurizer-%d.job" % n_choose, compress=True)


n_components = 5
tica = mixtape.tica.tICA(n_components=n_components, lag_time=lag_time)
pipeline = sklearn.pipeline.Pipeline([("features", tica_optimizer.featurizer), ('tica', tica)])
pipeline.fit(train)

pipeline.score(train), pipeline.score(test)
