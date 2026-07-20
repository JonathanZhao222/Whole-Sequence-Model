import numpy as np

# Amino acid order matching the HDF5 files
AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY')

PROPERTY_NAMES = [
    'hydrophobicity',   # Kyte-Doolittle
    'volume',           # residue volume (Å³)
    'charge',           # charge at pH 7
    'polarity',         # 1 = polar/charged, 0 = nonpolar
    'aromaticity',      # 1 = aromatic ring
    'helix_prop',       # Chou-Fasman P_alpha
    'sheet_prop',       # Chou-Fasman P_beta
]

# Rows in AA_ORDER: A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
PROPERTY_TABLE = np.array([
    #  hydro  volume  charge  polar  aromat  helix  sheet
    [  1.8,   88.6,    0.0,    0.0,   0.0,   1.45,  0.97],  # A
    [  2.5,  108.5,    0.0,    1.0,   0.0,   0.77,  1.30],  # C
    [ -3.5,  111.1,   -1.0,    1.0,   0.0,   0.98,  0.80],  # D
    [ -3.5,  138.4,   -1.0,    1.0,   0.0,   1.53,  0.26],  # E
    [  2.8,  189.9,    0.0,    0.0,   1.0,   1.12,  1.28],  # F
    [ -0.4,   60.1,    0.0,    0.0,   0.0,   0.53,  0.81],  # G
    [ -3.2,  153.2,    0.1,    1.0,   1.0,   1.24,  0.71],  # H
    [  4.5,  166.7,    0.0,    0.0,   0.0,   1.00,  1.60],  # I
    [ -3.9,  168.6,    1.0,    1.0,   0.0,   1.07,  0.74],  # K
    [  3.8,  166.7,    0.0,    0.0,   0.0,   1.34,  1.22],  # L
    [  1.9,  162.9,    0.0,    0.0,   0.0,   1.20,  1.67],  # M
    [ -3.5,  114.1,    0.0,    1.0,   0.0,   0.73,  0.65],  # N
    [ -1.6,  112.7,    0.0,    0.0,   0.0,   0.59,  0.62],  # P
    [ -3.5,  143.8,    0.0,    1.0,   0.0,   1.17,  1.23],  # Q
    [ -4.5,  173.4,    1.0,    1.0,   0.0,   0.79,  0.90],  # R
    [ -0.8,   89.0,    0.0,    1.0,   0.0,   0.79,  0.72],  # S
    [ -0.7,  116.1,    0.0,    1.0,   0.0,   0.82,  1.20],  # T
    [  4.2,  140.0,    0.0,    0.0,   0.0,   1.14,  1.65],  # V
    [ -0.9,  227.8,    0.0,    0.0,   1.0,   1.14,  1.19],  # W
    [ -1.3,  193.6,    0.0,    1.0,   1.0,   0.61,  1.29],  # Y
], dtype=np.float32)  # (20, 7)

# Background amino acid frequencies computed from 1000 human proteome proteins
# (551,521 residues from UP000005640_9606, AA order matches AA_ORDER above)
BACKGROUND_FREQ = np.array([
    0.0697, 0.0246, 0.0468, 0.0704, 0.0370, 0.0676, 0.0265, 0.0429,
    0.0571, 0.0980, 0.0206, 0.0363, 0.0645, 0.0475, 0.0556, 0.0831,
    0.0530, 0.0594, 0.0124, 0.0270,
], dtype=np.float32)  # (20,)
