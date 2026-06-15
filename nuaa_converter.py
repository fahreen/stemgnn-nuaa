import numpy as np, pandas as pd, glob
N_BINS = 256; WIN = np.hanning(N_BINS)
def load_iq(path):
    I, Q = [], []
    for ln in open(path, errors="ignore"):
        p = ln.replace("\r","").split("\t")
        if len(p) >= 2:
            try: i, q = float(p[0]), float(p[1])
            except ValueError: continue
            I.append(i); Q.append(q)
    return (np.array(I) + 1j*np.array(Q))[1:]      # drop header sample
def file_to_power_dB(path):
    iq = load_iq(path); iq = iq - iq.mean()
    n = len(iq)//N_BINS
    frames = iq[:n*N_BINS].reshape(n, N_BINS) * WIN
    power = (np.abs(np.fft.fft(frames, axis=1))**2).mean(axis=0)
    return 10*np.log10(power + 1e-12)
if __name__ == "__main__":
    files = sorted(glob.glob("path/to/nuaa/*.xls"))
    mat = np.vstack([file_to_power_dB(f) for f in files])
    pd.DataFrame(mat).to_csv("dataset/nuaa.csv", index=False)
