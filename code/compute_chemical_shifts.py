import pandas as pd
import mixtape.featurizer, mixtape.tica, mixtape.cluster, mixtape.markovstatemodel, mixtape.ghmm
import numpy as np
import mdtraj as md
from parameters import load_trajectories, build_full_featurizer
import sklearn.pipeline, sklearn.externals.joblib
import mixtape.utils

n_choose = 50
stride = 1
lag_time = 1
n_components = 2

trj0, trajectories, filenames = load_trajectories(stride=stride)


t0 = trajectories[0][0].join([t[0] for t in trajectories])
t1 = trajectories[0][-1].join([t[-1] for t in trajectories])

C0 = md.compute_chemical_shifts(t0)
C1 = md.compute_chemical_shifts(t1)

mu0 = C0.mean(1)
mu1 = C1.mean(1)

d = pd.read_csv("/home/kyleb/src/choderalab/FAHNVT/T4/data/nature_2011_SI_table4.csv")
d["resSeq"] = d.name.apply(lambda x: int(x[1:]))
d["resName"] = d.name.apply(lambda x: x[0:1])
d = d.set_index("resSeq")
d = d[["N", "H", "C", "CA", "HA"]]

d2 = {}
for (resSeq, row) in d.iterrows():
    for key in ["N", "H", "C", "CA", "HA"]:
        try:
            d2[(resSeq, key)] = float(row[key])
        except:
            pass

d2 = pd.Series(d2)
d2.index.names = ["resSeq", "name"]

err0 = (d2 - mu0).dropna()
err1 = (d2 - mu1).dropna()

rms0 = (err0 ** 2).reset_index().groupby("name").mean()[0]
rms1 = (err1 ** 2).reset_index().groupby("name").mean()[0]
