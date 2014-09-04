import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.datasets, mixtape.subset_featurizer, mixtape.feature_selection
import numpy as np
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 4500
n_choose = 10
lag_time = 1

trajectories = mixtape.datasets.alanine_dipeptide.fetch_alanine_dipeptide()["trajectories"]
train = trajectories[0::2]

featurizer = mixtape.subset_featurizer.guess_featurizers(trajectories[0], n_choose)

model = mixtape.tica.tICA(lag_time=lag_time, n_components=4)

tica_optimizer = mixtape.feature_selection.Optimizer(featurizer, model)
featurizer = tica_optimizer.optimize(n_iter, train)
