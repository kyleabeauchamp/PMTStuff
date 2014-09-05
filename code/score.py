import glob
import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel
import numpy as np
import mdtraj as md
from mixtape import ghmm, feature_selection, subset_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

n_iter = 1000
n_choose = 50
stride = 1
lag_time = 1
n_components = 2

filenames = glob.glob("./Trajectories/*.h5")
trajectories = [md.load(filename) for filename in filenames]

if len(trajectories) > 1:
    train = trajectories[0::2]
    test = trajectories[1::2]
else:
    train = [trajectories[0][0:trajectories[0].n_frames/2]]
    test = [trajectories[0][trajectories[0].n_frames/2:]]


featurizer = sklearn.externals.joblib.load("./featurizer-%d-%d.job" % (n_components, n_choose))
tica = mixtape.tica.tICA(lag_time=lag_time, n_components=n_components)
pipeline = sklearn.pipeline.Pipeline([("features", featurizer), ('tica', tica)])
pipeline.fit(train)
print(pipeline.score(train), pipeline.score(test))


pipeline.fit(trajectories)
X = pipeline.transform(trajectories)
q = np.concatenate(X)
hexbin(q[:, 0], q[:, 1], bins='log')
