import numpy as np
import time
from iodata import load_one
from gbasis.wrappers import from_iodata
from gbasis.integrals.overlap import overlap_integral
from gbasis.parsers import parse_gbs, make_contractions

z = 0.0
atoms = []
coords = []
for i in range(2):
    coords.append(np.array([0,0,z]))
    if i == 1:
        atoms.append("C")
    else:
        atoms.append("C")

    z += 1.5
coords = np.asarray(coords)

"""
with open("from_basissetexchange.gbs","r") as file:
    all_basis = file.read()
"""
print("hello")
filename = "data_631g.gbs"
#filename = "data_ccpVDZ.gbs"
#filename = "data_6311g.gbs"
#filename = "data_ccpVTZ.gbs"
global others
others = 0
all_basis_dict = parse_gbs(filename)
ovr = False
basis = make_contractions(all_basis_dict, atoms, coords, overlap = ovr)
stats=True
if ovr and stats:
    print(basis[0].ovr_mask)
    for i,item in enumerate(basis):
        if i == 0:
            print(i, item.ovr_mask)

import time
starrt = time.time()
output = overlap_integral(basis, transform="transform_mo_ao")
endd = time.time()
print("total time is {}".format(endd - starrt))