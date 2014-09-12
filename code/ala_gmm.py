import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.datasets, mixtape.subset_featurizer, mixtape.feature_selection
import numpy as np
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

# Copy paste from optimize ala


n_timescales = 4
n_states = 6
tica = mixtape.tica.tICA(n_components=n_components, lag_time=lag_time)
msm = mixtape.markovstatemodel.MarkovStateModel(n_timescales=n_timescales)
cluster = mixtape.cluster.GMM(n_components=n_states, covariance_type='full')
feature_pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica)])
cluster_pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("cluster", cluster)])
pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("cluster", cluster), ("msm", msm)])

pipeline.fit(train)
print(pipeline.score(train), pipeline.score(test))
X_all = feature_pipeline.transform(trajectories)
q = np.concatenate(X_all)

covars_ = cluster.covars_
covars_ = cluster.covars_.diagonal(axis1=1, axis2=2)

for i, j in [(0, 1)]:
    figure()
    title("%d" % n_states)
    hexbin(q[:,i], q[:, j], bins='log')
    errorbar(cluster.means_[:, i], cluster.means_[:, j], xerr=covars_[:,i] ** 0.5, yerr=covars_[:, j] ** 0.5, fmt='kx', linewidth=4)

offset = np.ones(2) * 0.05
for state in range(n_states):    
    plt.annotate("%d" % state, cluster.means_[state, 0:2] + offset, fontsize='x-large')



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


figure()
for i in range(n_states):
    ind = states == i
    plot(phi[ind], psi[ind], '.')

xlim(-180, 180)
ylim(-180, 180)
