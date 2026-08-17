import numpy as np
import matplotlib.pyplot as plt

# ---- settings ----
dump_file = input("Path to dump file: ")
label = input("Label (e.g. 22, pruned_1): ")
dt_ps = 0.001          # 1 fs timestep, expressed in ps
jump_cut = 2.3         # Si-Si bond length, Angstrom
voxel = 0.5            # Angstrom, drawn on the histogram
# ------------------

# --- read the dump ---
positions = []   # real coordinates, one (n_atoms, 3) array per frame
boxes = []       # (Lx, Ly, Lz) per frame, since NPT lets the box change
steps = []

with open(dump_file) as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if "ITEM: TIMESTEP" in lines[i]:
        steps.append(int(lines[i + 1]))
        n = int(lines[i + 3])

        lo_hi = [list(map(float, lines[i + 5 + k].split()[:2])) for k in range(3)]
        origin = np.array([lo for lo, hi in lo_hi])
        L = np.array([hi - lo for lo, hi in lo_hi])
        boxes.append(L)

        block = np.array([list(map(float, lines[i + 9 + k].split()))
                          for k in range(n)])
        block = block[block[:, 0].argsort()]      # sort by id, order varies per frame
        positions.append(origin + block[:, 2:5] * L)

        i += 9 + n
    else:
        i += 1

pos = np.array(positions)            # (n_frames, n_atoms, 3)
box = np.array(boxes)                # (n_frames, 3)
steps = np.array(steps)
n_frames, n_atoms, _ = pos.shape
print(f"{n_frames} frames, {n_atoms} atoms")

# --- unwrap: undo jumps across the periodic boundary ---
d = np.diff(pos, axis=0)
Lmid = box[:-1][:, None, :]
d -= Lmid * np.round(d / Lmid)                    # minimum image
r = np.zeros_like(pos)
r[0] = pos[0]
r[1:] = pos[0] + np.cumsum(d, axis=0)

# --- remove overall drift of the system ---
r -= r.mean(axis=1, keepdims=True)

time_ps = (steps - steps[0]) * dt_ps

# --- 1. vibration amplitude ---
site = r.mean(axis=0, keepdims=True)              # average position of each atom
dev = np.linalg.norm(r - site, axis=2)            # (n_frames, n_atoms)

print(f"\nmean amplitude = {dev.mean():.3f} A")
print(f"rms amplitude  = {np.sqrt((dev ** 2).mean()):.3f} A")
print(f"max amplitude  = {dev.max():.3f} A")
print(f"fraction above {voxel} A = {(dev > voxel).mean():.2e}")

plt.figure(figsize=(6, 4))
plt.hist(dev.ravel(), bins=80,
         weights=np.ones(dev.size) / dev.size, color="steelblue")
plt.yscale("log")
plt.axvline(voxel, color="red", ls="--", label=f"voxel = {voxel} A")
plt.axvline(dev.mean(), color="black", ls=":", label=f"mean = {dev.mean():.2f} A")
plt.xlabel("displacement from average position (A)")
plt.ylabel("fraction of counts")
plt.title(f"Vibration amplitude, {label}")
plt.legend()
plt.tight_layout()
plt.savefig(f"vibration_amplitude_{label}.png", dpi=150)
print(f"saved vibration_amplitude_{label}.png")

# --- 2. jumps: displacement from the starting position ---
disp = np.linalg.norm(r - r[0], axis=2)           # (n_frames, n_atoms)
print(f"\nmax displacement from t=0: {disp.max():.3f} A")

hits = np.argwhere(disp > jump_cut)
if len(hits) == 0:
    print(f"no atom moves more than {jump_cut} A from its start")
else:
    atoms = np.unique(hits[:, 1])
    print(f"{len(atoms)} atom(s) cross {jump_cut} A")
    for a in atoms:
        first = hits[hits[:, 1] == a][0, 0]
        print(f"  id {a + 1}: first crossing at {time_ps[first]:.1f} ps, "
              f"max {disp[:, a].max():.2f} A, ends at {disp[-1, a]:.2f} A")

# --- 3. jumps that reverse: displacement over short windows ---
print()
for w in [10, 50, 100]:
    if w >= n_frames:
        continue
    win = np.linalg.norm(r[w:] - r[:-w], axis=2)
    print(f"window {time_ps[w] - time_ps[0]:.1f} ps: max displacement {win.max():.3f} A")

plt.figure(figsize=(6, 4))
plt.plot(time_ps, disp.max(axis=1), color="darkred")
plt.axhline(jump_cut, color="black", ls="--", label=f"{jump_cut} A")
plt.xlabel("time (ps)")
plt.ylabel("largest displacement from t=0 (A)")
plt.title(f"Maximum displacement, {label}")
plt.legend()
plt.tight_layout()
plt.savefig(f"max_displacement_{label}.png", dpi=150)
print(f"saved max_displacement_{label}.png")