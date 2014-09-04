import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.datasets, mixtape.subset_featurizer, mixtape.feature_selection
import numpy as np
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 2500
n_choose = 10
lag_time = 1

trajectories = mixtape.datasets.alanine_dipeptide.fetch_alanine_dipeptide()["trajectories"]
train = trajectories[0::2]

featurizer = sklearn.externals.joblib.load("./featurizer-%d.job" % n_choose)


n_components = 2
n_states = 4
tica = mixtape.tica.tICA(n_components=n_components, lag_time=lag_time)
subsampler = mixtape.utils.Subsampler(lag_time=lag_time)
msm = mixtape.markovstatemodel.MarkovStateModel()
cluster = mixtape.cluster.GMM(n_components=n_states, covariance_type='full')
feature_pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica)])
cluster_pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("cluster", cluster)])
pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("subsampler", subsampler), ("cluster", cluster), ("msm", msm)])

pipeline.fit(train)
X_all = feature_pipeline.transform(trajectories)
q = np.concatenate(X_all)



dih_featurizer = mixtape.featurizer.DihedralFeaturizer(["phi", "psi"], sincos=False)
X = dih_featurizer.transform(trajectories)
phi, psi = np.concatenate(X).T * 180 / np.pi

hexbin(phi, psi)

states = pipeline.transform(trajectories)
states = np.concatenate(states)

for i in range(n_states):
    figure()
    ind = states == i
    plot(phi[ind], psi[ind], "*")
    xlim(-180, 180)
    ylim(-180, 180)
