import ffmpy
import argparse, sys
import os

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--dir', type = str, default = './data/video/')
args = parser.parse_args()

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(args.dir) if isfile(join(args.dir, f))]

only_mp4 = []
for f in onlyfiles:
    video = f[:-4]
    if video + '.gif' in onlyfiles:
        continue
    only_mp4.append(f)

for f in only_mp4:
    ff = ffmpy.FFmpeg(
        inputs = {args.dir + f : None},
        outputs = {args.dir + f[:-4] + '.gif' : '-vf "scale=640:-1:flags=lanczos"'})
    
    ff.run()