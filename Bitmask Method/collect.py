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
    if len(seed) > max_len:
        seed = seed[:max_len]
    return seed


def xor(seed, mutation):
    return bytes([_a ^ _b for _a, _b in zip(pad(seed), pad(mutation))])


def vectorize(seed):
    shape = np.zeros(shape=(max_len, bytesize))
    for byte_pos, byte in enumerate(seed):
        bits = bin(byte)[2:].zfill(8)
        for n, bit in enumerate(bits):
            if bit == '1':
                shape[byte_pos, n] = 1.
    return shape.reshape((shape.shape[0]//8,bytesize*8))


def get_coverage(mut, depth):
    if depth <= 0:
        return mut.coverage
    cov = any([get_coverage(mut, depth-1) for mut in all_inputs if mut.src and mut.id in mut.src])
    return cov if depth == default_depth else mut.coverage or cov # If we take the root mutation's coverage, ~4 times more inputs appear.

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
        self.op_prop = {k: props[k] for k in props.keys() if k not in ['id','src','time', 'op']}
        self.coverage = '+cov' in self.op_prop.keys()
        self.content = open(name, 'rb').read()


def get_lib_data(lib):
    files = os.listdir(lib)

    all_inputs = [mutation(lib + '/' + input) for input in files if len(input.split(',')) >= 4]

    increase_inputs = [x for x in all_inputs if x.coverage == True]

    all_inputs_ids = {str(mut.id) : mut.content for mut in all_inputs}

    x_data = []
    y_data = []

    for input in increase_inputs:
        for src_id in input.src.split('+'):
            if src_id in all_inputs_ids:
                x_data.append(vectorize(pad(all_inputs_ids[src_id])))
                y_data.append(vectorize(xor(all_inputs_ids[src_id], input.content)))

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


saveToFile(x_data, y_data, "mutations_bytemask")
    


