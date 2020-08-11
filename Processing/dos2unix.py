#!/usr/bin/env python
#from stackoverflow answer https://stackoverflow.com/questions/45368255/error-in-loading-pickle
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py 
"""
import os

os.chdir(".\\Processing")
print(os.path.abspath(os.path.curdir))

original = "sdcalib.rmap.full.camera.pickle"
destination = "sdcalib.rmap.full.camera_unix.pickle"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))