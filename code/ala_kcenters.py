import mdtraj as md
import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.datasets, mixtape.subset_featurizer, mixtape.feature_selection
import numpy as np
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils


trajectories = mixtape.datasets.alanine_dipeptide.fetch_alanine_dipeptide()["trajectories"]
train = trajectories[0::2]
test = trajectories[1::2]

n_timescales = 4
n_clusters = 100

clusterer = mixtape.cluster.KCenters(n_clusters=n_clusters, metric=md.rmsd)
msm = mixtape.markovstatemodel.MarkovStateModel(n_timescales=n_timescales)

pipeline = sklearn.pipeline.Pipeline([("clusterer", clusterer), ("msm", msm)])
assignments = pipeline.fit_transform(train)

msm.timescales_
print(pipeline.score(train), pipeline.score(test))
