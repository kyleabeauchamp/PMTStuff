import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel
import numpy as np
import mdtraj as md
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 50
n_choose = 40
stride = 4
lag_time = 1

trj0, trajectories, filenames = load_trajectories(stride=stride)

train = trajectories[0:100]
test = trajectories[100:]

d_min = 0.25
cluster = mixtape.cluster.RegularSpatial(d_min=d_min, metric=md.rmsd)

msm = mixtape.markovstatemodel.MarkovStateModel(n_timescales=3)
pipeline = sklearn.pipeline.Pipeline([("cluster", cluster), ("msm", msm)])

pipeline.fit(train)
pipeline.score(train), pipeline.score(test)


