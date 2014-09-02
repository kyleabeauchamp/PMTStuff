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

n_components_list = [5]
n_states_list = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175]
train_scores = np.zeros((len(n_components_list), len(n_states_list)))
test_scores = np.zeros((len(n_components_list), len(n_states_list)))

for i, n_components in enumerate(n_components_list):
    for j, n_states in enumerate(n_states_list):
        print(n_components, n_states)
        tica = mixtape.tica.tICA(n_components=n_components, lag_time=lag_time)
        subsampler = mixtape.utils.Subsampler(lag_time=lag_time)
        msm = mixtape.markovstatemodel.MarkovStateModel(n_timescales=n_components)
        cluster = mixtape.cluster.KMeans(n_states)
        pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica), ("subsampler", subsampler), ("cluster", cluster), ("msm", msm)])
        pipeline.fit(train)
        train_scores[i, j] = pipeline.score(train)
        test_scores[i, j] = pipeline.score(test)


plot(n_states_list, train_scores.T, 'o', label="train")
plot(n_states_list, test_scores.T, 'o', label="test")

xlabel("n_states")
ylabel("Score")
title("tICA KMeans SETD2")
legend(loc=0)
savefig("/home/kyleb/src/kyleabeauchamp/MixtapeTalk/figures/SETD2_tICA_KMeans.png")

