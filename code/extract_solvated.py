import os
import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel
import numpy as np
import mdtraj as md
from mixtape import ghmm, selector, subset_featurizer, selector
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

stride = 1
lag_time = 1

trj0, trajectories, filenames = load_trajectories(stride=stride)
allatom_filenames = [os.path.join("./full_Trajectories/", os.path.split(f)[-1]) for f in filenames]

# Load model

ind, mu = model.draw_centroids(X_all)
samples = mixtape.utils.map_drawn_samples(ind, allatom_filenames)

for k, t in enumerate(samples):
    t.save("solvated_pdbs/state%d-mean.pdb" % k)
