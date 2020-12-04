from os import listdir
from os.path import isfile, join
from pathlib import Path

from_path='/home/victor/MALIS-PDB'

A=[0 for _ in range(5)]

for f in [f for f in listdir(from_path) if isfile(join(from_path, f))]:
    id=f.split('.')[0]
    
    for i in range(5):
        if id in C[i][1].split(' '):
            A[i]+=1

print(A)