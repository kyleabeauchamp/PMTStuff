import mixtape.featurizer, mixtape.tica
import numpy as np
import mdtraj as md
from mixtape import ghmm, selector, subset_featurizer, selector
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils, mixtape.cluster


n_iter = 50
n_choose = 40
stride = 1
lag_time = 1
n_macro = 5

trj0, trajectories, filenames = load_trajectories(stride=stride)

clusterer = mixtape.cluster.KCenters(n_clusters=400, metric=md.rmsd)
pcca = mixtape.lumping.PCCAPlus(n_macro)
macro_msm = mixtape.markovstatemodel.MarkovStateModel()


pipeline = sklearn.pipeline.Pipeline([("clusterer", clusterer), ("pcca", pcca), ("macro_msm", macro_msm)])
macro_assignments = pipeline.fit_transform(trajectories)

ind = mixtape.markovstatemodel.draw_samples(macro_assignments, 3)
samples = mixtape.utils.map_drawn_samples(ind, trajectories)

for i in range(n_macro):
    for k, t in enumerate(samples[i]):
        t.save("msmstate%d-%d.pdb" % (i, k))


rho = np.array([[mean(ass==i) for i in range(5)] for ass in macro_assignments])
