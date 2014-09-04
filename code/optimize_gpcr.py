import glob
import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel
import numpy as np
import mdtraj as md
from mixtape import ghmm, subset_featurizer, feature_selection
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 1000

n_choose = 100
stride = 1
lag_time = 1

trj0 = md.load("./system.subset.pdb")
filenames = glob.glob("./Trajectories/*.lh5")

#trajectories = [md.load(filename) for filename in filenames[::50]]
train = [md.load(filename) for filename in filenames[::100]]
#train = [md.load(filename) for filename in filenames[::2]]
#train = train[0::10]

featurizer = subset_featurizer.guess_featurizers(trj0, n_choose)  # Doesn't work right now, too many features need to re-optimize later.
#featurizer = sklearn.externals.joblib.load("./featurizer-%d.job" % n_choose)

tica_optimizer = mixtape.feature_selection.TICAOptimizer(featurizer, lag_time=lag_time)
tica_optimizer.optimize(n_iter, train)

#sklearn.externals.joblib.dump(tica_optimizer.featurizer, "./featurizer-%d.job" % n_choose, compress=True)


n_components = 5
tica = mixtape.tica.tICA(n_components=n_components, lag_time=lag_time)
pipeline = sklearn.pipeline.Pipeline([("features", tica_optimizer.featurizer), ('tica', tica)])
pipeline.fit(train)

pipeline.score(train), pipeline.score(test)
