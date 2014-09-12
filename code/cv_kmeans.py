import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.ghmm
import numpy as np
import mdtraj as md
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

n_choose = 50
stride = 1
lag_time = 1
n_components = 2

trj0, trajectories, filenames = load_trajectories(stride=stride)

train = trajectories[0::2]
test = trajectories[1::2]

featurizer = sklearn.externals.joblib.load("./featurizer-%d-%d.job" % (n_components, n_choose))


for n_states in [10, 20, 30, 40, 50]:
    n_components = n_components
    tica = mixtape.tica.tICA(n_components=n_components, lag_time=lag_time)
    msm = mixtape.markovstatemodel.MarkovStateModel(n_timescales=5)
    cluster = mixtape.cluster.KMeans(n_clusters=n_states)
    pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("cluster", cluster), ("msm", msm)])
    pipeline.fit(train)
    print(pipeline.score(train), pipeline.score(test))
    pipeline.fit(trajectories)
    print(msm.timescales_)
