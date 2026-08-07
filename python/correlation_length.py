import os
import pandas as pd
import numpy as np

# Read output.txt
def read_output_file(file_path):
    df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None)
    print("Header: "+str(df.head()))
    return df

def find_peaks(data):
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)

file_path = input("Enter the path to the output.txt file: ")
if not os.path.isfile(file_path):
    print(f"Error: {file_path} does not exist.")
    exit(1)
else:
    df = read_output_file(file_path)
    radius = df.iloc[:, 0].values
    gpots = [df.iloc[:, 1].values, df.iloc[:, 2].values, df.iloc[:, 3].values, df.iloc[:, 4].values]
    print("Radius: "+str(radius)+"\nGpots: "+str(gpots))

labels = ['22', 'pruned_1', 'pruned_2', 'pruned_3']
xis = []
for lbl, g in zip(labels, gpots):
    h = radius * np.abs(g - 1)

    troughs = find_peaks(-(g - 1))
    peaks = find_peaks(g - 1)
    troughs = find_peaks(-(g - 1))

    peaks = peaks[g[peaks] > 1]
    troughs = troughs[g[troughs] < 1]

    idx = np.sort(np.concatenate([peaks, troughs]))
    
    idx = idx[(radius[idx] > 3) & (radius[idx] < 8.3)] # Keep only indices where 3 < radius < 8.3
    
    slope, intercept = np.polyfit(radius[idx], np.log(h[idx]), 1) # Slope, intercept = np.polyfit(radius[idx], np.log(h[idx]), 1)

    xi = -1 / slope

    xis.append(xi)
    
    fit = np.polyval([slope, intercept], radius[idx])
    y = np.log(h[idx])
    r2 = 1 - np.sum((y - fit)**2) / np.sum((y - y.mean())**2)
    print(lbl, "n =", len(idx), "xi =", round(xi, 2), "R2 =", round(r2, 3))
print("Correlation lengths (xi): "+str(xis))