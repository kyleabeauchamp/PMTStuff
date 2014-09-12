import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.ghmm
import numpy as np
import mdtraj as md
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

n_choose = 50
n_components = 2

t = md.load("./system.subset.pdb")

featurizer_old = sklearn.externals.joblib.load("./featurizer-%d-%d.job" % (n_components, n_choose))
featurizer = mixtape.subset_featurizer.guess_featurizers(t, n_choose)

featurizer.subsets = featurizer_old.subsets
featurizer.transform([t])  # Check that it can transform
f2 = sklearn.clone(featurizer)  # Check that it can clone.

sklearn.externals.joblib.dump(featurizer_old, "./featurizer-%d-%d.job.old" % (n_components, n_choose), compress=True)
sklearn.externals.joblib.dump(featurizer, "./featurizer-%d-%d.job" % (n_components, n_choose), compress=True)
