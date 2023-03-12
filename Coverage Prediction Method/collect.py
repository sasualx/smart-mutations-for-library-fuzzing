import sys
import csv
import os
import numpy as np

all_inputs = []
default_depth = 5 # The difference between 5 and 15 is 0.1%
max_len = 5000
bytesize = 8

def pad(seed):
    if len(seed) < max_len:
        seed = seed + b'\x00' * (max_len - len(seed))
    return seed


def xor(seed, mutation):
    return bytes([_a ^ _b for _a, _b in zip(pad(seed), pad(mutation))])


def get_coverage(mut, depth, inputs):
    if depth <= 0:
        return 1 if mut.coverage else 0
    cov = sum([get_coverage(lower_mut, depth-1, inputs) for lower_mut in inputs if mut.src and lower_mut.id in mut.src])
    return (1 if mut.coverage else 0) + cov


def pad_vectorize(bytestream):
    return np.pad(np.frombuffer(bytestream, dtype=np.uint8), (0, max_len - len(bytestream)), mode='constant')\
        .reshape((max_len, 1))

def saveToFile(sample_x, sample_y, output_file):
    np.savez_compressed(output_file, x=sample_x, y=sample_y)
    print("[+] Dataset stored in: " + output_file + ".npz")

    return 0

class mutation:
    def __init__(self, name):
        file = name.split('/')[-1]
        props = {prop.split(":")[0] : prop.split(":")[1] if len(prop.split(':')) > 1 else True for prop in file.split(',')}
        self.id = props['id']
        self.src = props['src'] if 'src' in props.keys() else None
        self.time = props['time']
        self.op = props['op'] if 'op' in props.keys() else None
        self.op_prop = {k: props[k] for k in props.keys() if k not in ['id', 'src', 'time', 'op']}
        self.coverage = '+cov' in self.op_prop.keys()
        self.content = open(name, 'rb').read()


def get_lib_data(lib):
    files = os.listdir(lib)
    all_inputs = [mutation(lib + '/' + input) for input in files if len(input.split(',')) >= 4]

    all_inputs_ids = {str(mut.id): mut.content for mut in all_inputs}
    x_data = []
    y_data = []

    for i in all_inputs:
        if i.src:
            for src in i.src.split('+'):
                if src in all_inputs_ids.keys() and len(all_inputs_ids[src]) <= max_len and len(all_inputs_ids[i.id]) <= max_len:
                    x_data.append(pad_vectorize(xor(all_inputs_ids[src], all_inputs_ids[i.id])))
                    y_data.append(get_coverage(i, default_depth, all_inputs))
    return x_data, y_data


if len(sys.argv) < 2:
    print("You need to give a path as an argument")
    sys.exit()

libs = sys.argv[1:]
x_data = []
y_data = []

for lib in libs:
    x_data_new, y_data_new = get_lib_data(lib)
    x_data += x_data_new
    y_data += y_data_new


saveToFile(x_data, y_data, "mutations_coverage_prediction")