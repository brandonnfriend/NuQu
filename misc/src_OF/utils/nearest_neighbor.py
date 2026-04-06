import numpy as numpy


def get_NN_from_site_1D(site_id:int, array_length:int):
    if site_id == 0:
        return [1]
    elif site_id == array_length-1:
        return [array_length-2]
    else:
        return [site_id-1, site_id+1]

def get_NN_pairs_1D(array_length:int):
    pairs = []
    for i in range(array_length-1):
        pairs.append((i, i+1))
    return pairs



