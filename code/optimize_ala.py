import mdtraj as md
import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.datasets, mixtape.subset_featurizer, mixtape.feature_selection
import numpy as np
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


n_iter = 2500
n_choose = 10
lag_time = 1
n_components = 4

trajectories = mixtape.datasets.alanine_dipeptide.fetch_alanine_dipeptide()["trajectories"]
train = trajectories[0::2]
test = trajectories[1::2]

clusterer = mixtape.cluster.KCenters(n_clusters=10, metric=md.rmsd)
clusterer.fit(train)

featurizer = mixtape.subset_featurizer.guess_featurizers(trajectories[0][0], n_choose, clusterer.cluster_centers_)
featurizer.transformer_list[0][1].subset = np.array([], 'int')
featurizer.transformer_list[-1][1].subset = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'int')

model = mixtape.tica.tICA(lag_time=lag_time, n_components=n_components)
tica_optimizer = mixtape.feature_selection.Optimizer(featurizer, model, n_iter)

featurizer = tica_optimizer.optimize(train)


tica = mixtape.tica.tICA(lag_time=lag_time, n_components=n_components)
pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica)])
pipeline.fit(train)
print(pipeline.score(train), pipeline.score(test))
