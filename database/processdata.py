# deal with imports
import pandas as pd
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load data

fname = 'ame2020_raw'

with open(fname, 'r') as f:
    lines = f.readlines()

lines = lines[3:]
datalen = len(lines)

# z & n
zn_lines = [line[6:14] for line in lines]

# masses w/ uncertainty
mu_lines = [line[106:-1] for line in lines]

values = np.zeros((datalen, 4))

for i, mu_line in enumerate(mu_lines):
    zn_line = zn_lines[i]
    sline = (zn_line+' '+mu_line).split()
    z, n, M, m, u = sline
    if m[-1] == '#' and u[-1] == '#':
        m, u = m[:-1], u[:-1]
    M, m, u = float(M), float(m), float(u)
    values[i] = (z, n, M+m/1e6, u/1e6)

values[48, -1] += 1e-12

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# set up features

# def constants
Ns = [2, 8, 20, 28, 50, 82, 126, 184, 258]
Nm = [(Ns[i+1]+Ns[i])//2 for i in range(8)]


# set up features
def isospin_asym(z, n): return (n-z)/(n+z)
def parity(i): return 1 if (i % 2 == 1) else 0


def valence(n):
    if n < 5:
        return n - 2
    for j, s in enumerate(Ns):
        S = Ns[j+1]
        if n < S and n >= s:
            m = (s+S) // 2
            if n < m:
                return n - s
            elif n >= m:
                return n - S

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# save data


# set up entry
def make_entry(row):
    n, z, mass, u = row
    pasym = isospin_asym(z, n)
    pz, pn = parity(z), parity(n)
    vz, vn = valence(z),  valence(n)
    return [z, n, z+n, pasym, pz, pn, vz, vn, mass, u]


entries = []
for row in values:
    entries.append(make_entry(row))

# make dataframe
df = pd.DataFrame(columns=['Z', 'N', 'A', 'Pasym', 'ZPAR', 'NPAR', 'VZ', 'VN',
                           'MASS', 'U'], data=entries)
df.to_csv('ame2020.csv')
