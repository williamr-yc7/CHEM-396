3 pruned potentials in order by increasing cost and accuracy
2 potentials (level 22 and level 26) in the original_for_melt which are fitted explicility to be more stable in melts

Pruned may be unstable in melts. In this case you proceed in two steps, as is often done in the literature:

Melting: use one of the level-based potentials (e.g., level 22). The melting is inexpensive even for a large box, a few tens of picoseconds are sufficient. The system can be heated up to 4500–5000 K and then rapidly cooled (10–25 K/ps) down to 3500–3000 K.

Cooling: for the slow cooling stage, they can switch to the pruned potential.