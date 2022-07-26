import geopandas as gpd
import matplotlib.pyplot as plt
# For CPU calculation
import numpy as np
# For GPU calculation
import cupy as cp
import time

from setting import MAP_DIR, GPU


# Using GPU for calulate length
def length_cp(row):
    print("length_cp")
    a = time.time()
    val = cp.array(row["geometry"].xy)
    # formula for calulate length
    length = cp.sum(
        cp.sqrt(
            cp.square(cp.diff(val[0], 1)) + cp.square(cp.diff(val[1], 1))
        )
    )
    print("Second:", time.time() - a)
    print("----")
    return float(length)


# Using CPU for calculate length
def length_np(row):
    print("length_np")
    a = time.time()
    val = np.array(row["geometry"].xy)
    # formula for calulate length
    length = np.sum(
        np.sqrt(
            np.square(np.diff(val[0], 1)) + np.square(np.diff(val[1], 1))
        )
    )
    print("Second:", time.time() - a)
    print("----")
    return length


# Using special dll of Shapely lib
def length(row):
    print("length")
    a = time.time()
    val = row["geometry"].length
    print("Second:", time.time() - a)
    print("----")
    return val


# Function for parse,show and calulate length for all coastline
def read_map_and_calc_length():
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    df = gpd.read_file(MAP_DIR.joinpath('ne_10m_coastline.shp'))
    # Managing rows for calculation
    df = df.iloc[3:5]
    a = time.time()
    if GPU:
        # Calculate using GPU
        df["length_cp"] = df.apply(length_cp, axis=1)
    else:
        # Calculate using CPU
        df["length_np"] = df.apply(length_np, axis=1)
    b = time.time()
    df["length"] = df.apply(length, axis=1)
    print(df)
    df = df.set_geometry('geometry')
    df.plot(ax=ax)
    print("Second of all calc.:", b - a)
    # Show coastlines
    # plt.show()
