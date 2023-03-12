#!/usr/bin/python3

# Query Model function for afl-fuzz.c

import os
import numpy as np
from keras.models import load_model
import random


model = load_model("../../best_model_lstm_checkpoint.h5")
max_len = 5000

def xor(seed, mutation):
    if len(seed) < len(mutation):
        seed = seed + b'\x00' * (len(mutation) - len(seed))
    else:
        mutation = mutation + b'\x00' * (len(seed) - len(mutation))
    return bytes([_a ^ _b for _a, _b in zip(seed, mutation)])

def diff(input_file, seed):

    input = open('out/default/queue/' + [x for x in os.listdir('out/default/queue') if 'id:' + input_file in x][0], 'rb').read()
    diff = xor(input, seed)

    with open("test.txt", 'ab') as fd:
        fd.write(b"\n" + bytes(input) + b" + " + bytes(seed) + b" : " + diff)

    return diff

def pad_vectorize(bytestream):
    if len(bytestream) > max_len:
        bytestream = bytestream[:max_len]
    return np.pad(np.frombuffer(bytestream, dtype=np.uint8), (0, max_len - len(bytestream)), mode='constant')

def is_useful(source, new):
    
    try:
        if random.random() > 0.5:
            input = pad_vectorize(open('out/default/queue/' + [x for x in os.listdir('out/default/queue') if 'id:' + source in x][0], 'rb').read())
            res = model.predict(np.bitwise_xor(input, pad_vectorize(new)).reshape(1, max_len//40, 40)).flatten()[-1] > 2.5
            return res
    except Exception as e:
        print(e) 
    
    return True
