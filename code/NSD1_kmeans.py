import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.ghmm
import numpy as np
import mdtraj as md
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

n_choose = 100
stride = 1
lag_time = 1

trj0, trajectories, filenames = load_trajectories(stride=stride)

train = trajectories[0::2]
test = trajectories[1::2]

featurizer = sklearn.externals.joblib.load("./featurizer-%d.job" % n_choose)

n_components = 3
n_states = 100
tica = mixtape.tica.tICA(n_components=n_components, lag_time=lag_time)
subsampler = mixtape.utils.Subsampler(lag_time=lag_time)
msm = mixtape.markovstatemodel.MarkovStateModel(n_timescales=n_components)
cluster = mixtape.cluster.KMeans(n_states)
feature_pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica)])
cluster_pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("cluster", cluster)])
pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("subsampler", subsampler), ("cluster", cluster), ("msm", msm)])

pipeline.fit(train)
pipeline.score(train), pipeline.score(test)


X_all = feature_pipeline.transform(trajectories)
q = np.concatenate(X_all)
covars_ = cluster.covars_
covars_ = cluster.covars_.diagonal(axis1=1, axis2=2)

for i, j in [(0, 1)]:
    figure()
    hexbin(q[:,i], q[:, j], bins='log')
    errorbar(cluster.means_[:, i], cluster.means_[:, j], xerr=covars_[:,i] ** 0.5, yerr=covars_[:, j] ** 0.5, fmt='kx', linewidth=4)


states = cluster_pipeline.transform(trajectories)
ind = msm.draw_samples(states, 3)
samples = mixtape.utils.map_drawn_samples(ind, trajectories)

for i in range(n_states):
    for k, t in enumerate(samples[i]):
        t.save("pdbs/state%d-%d.pdb" % (i, k))
