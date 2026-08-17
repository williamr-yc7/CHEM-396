import numpy as np
d = np.loadtxt("OVITO/crystallinity_22.txt", skiprows=1)
other = d[:, 2]
hexd = d[:, 7]   # IdentifyDiamond.counts.HEX_DIAMOND
print("mean classified:", (216 - other).mean())
print("mean crystallinity %:", 100 * (216 - other).mean() / 216)
print("frames with any:", np.mean(other < 216) * 100, "%")
print("frames with hexagonal:", np.sum(hexd > 0))

import itertools
on = d[:, 2] < 216
runs = [(k, len(list(g))) for k, g in itertools.groupby(on)]
print("longest on:", max(l for k, l in runs if k))
print("mean on:", np.mean([l for k, l in runs if k]))
print("number of runs:", len(runs))