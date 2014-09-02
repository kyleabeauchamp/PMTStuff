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
featurizer = sklearn.externals.joblib.load("./featurizer-%d.job" % n_choose)

train = trajectories[0::2]
test = trajectories[1::2]


n_components = 3
tica = mixtape.tica.tICA(n_components=n_components, lag_time=lag_time)
subsampler = mixtape.utils.Subsampler(lag_time=lag_time)
pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("subsampler", subsampler)])

X_all = pipeline.fit_transform(trajectories)
q = np.concatenate(X_all)

n_states = 3
model = mixtape.ghmm.GaussianFusionHMM(n_states, n_components, fusion_prior=0., init_algo='GMM')
model.fit(X_all)
model.score(X_all)


for i, j in [(0, 1)]:
    figure()
    hexbin(q[:,i], q[:, j], bins='log')
    errorbar(model.means_[:, i], model.means_[:, j], xerr=model.vars_[:,i] ** 0.5, yerr=model.vars_[:, j] ** 0.5, fmt='kx', linewidth=4)


figure(1)
i, j = 0, 1
offset = np.ones(2) * 0.05
for state in range(n_states):    
    plt.annotate("%d" % state, model.means_[state, 0:2] + offset, fontsize='x-large')






ind, mu = model.draw_centroids(X_all)
samples = mixtape.utils.map_drawn_samples(ind, trajectories)

for k, t in enumerate(samples):
    t.save("pdbs/state%d-mean.pdb" % k)


ind = model.draw_samples(X_all, 3)
samples = mixtape.utils.map_drawn_samples(ind, trajectories)

for i in range(n_states):
    for k, t in enumerate(samples[i]):
        t.save("pdbs/state%d-%d.pdb" % (i, k))
