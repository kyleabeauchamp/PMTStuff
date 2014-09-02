import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.ghmm
import numpy as np
import mdtraj as md
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

stride = 1
lag_time = 1

trj0, trajectories, filenames = load_trajectories(stride=stride)

train = trajectories[0::2]
test = trajectories[1::2]

#n_states_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400]

n_states_list = range(5, 50)
train_scores = np.zeros(len(n_states_list))
test_scores = np.zeros(len(n_states_list))

for j, n_states in enumerate(n_states_list):
    print(n_components, n_states)
    subsampler = mixtape.utils.Subsampler(lag_time=lag_time)
    msm = mixtape.markovstatemodel.MarkovStateModel(n_timescales=n_components)
    cluster = mixtape.cluster.KCenters(n_states, metric=md.rmsd)
    pipeline = sklearn.pipeline.Pipeline([("subsampler", subsampler), ("cluster", cluster), ("msm", msm)])
    pipeline.fit(train)
    train_scores[j] = pipeline.score(train)
    test_scores[j] = pipeline.score(test)


plot(n_states_list, train_scores, 'o', label="train")
plot(n_states_list, test_scores, 'o', label="test")


xlabel("n_states")
ylabel("Score")
title("KCenters SETD2")
legend(loc=0)
savefig("/home/kyleb/src/kyleabeauchamp/MixtapeTalk/figures/SETD2_kcenters.png")

