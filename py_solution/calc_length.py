import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import time

from setting import MAP_DIR


def length_np(row):
    print("length_np index:", row.index)
    a = time.time()
    val = np.array(row["geometry"].xy)
    length = np.sum(
        np.sqrt(
            np.square(np.diff(val[0], 1)) + np.square(np.diff(val[1], 1))
        )
    )
    print("Second:", time.time() - a)
    print("----")
    return length


def length(row):
    print("length index:", row.index)
    a = time.time()
    val = row["geometry"].length
    print("Second:", time.time() - a)
    print("----")
    return val


def read_map_and_calc_length():
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    df = gpd.read_file(MAP_DIR.joinpath('ne_10m_coastline.shp'))
    df = df.iloc[0:2]
    df["length"] = df.apply(length, axis=1)
    df["length_np"] = df.apply(length_np, axis=1)
    print(df)
    df = df.set_geometry('geometry')
    df.plot(ax=ax)
    # plt.show()
