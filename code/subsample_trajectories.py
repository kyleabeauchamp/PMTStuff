import os
import mdtraj as md
import glob

stride = 1
min_num_frames = 400
filenames = glob.glob("/home/kyleb/dat/FAH/munged/protein/10470/*.h5")

for k, filename in enumerate(filenames):
    print(filename)
    trj = md.formats.HDF5TrajectoryFile(filename, mode='r')
    n_frames = len(trj)
    trj.close()
    if n_frames >= min_num_frames:
        out_filename = os.path.join("./Trajectories/", os.path.basename(filename))
        md.load(filename, stride=stride).save(out_filename)
        
