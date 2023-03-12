#!/usr/bin/python3

# Query Model function for afl-fuzz.c

import os
import numpy as np
from keras.models import load_model
import random

model = load_model("../../best_model_lstm_checkpoint.h5")
max_len = 5000 #30720, 20480
bytesize = 8

def query_model(seed):
    # Query model for the prediction
    predictions = model.predict(seed)
    # Determine bytemask from prediction output
    bytemask = get_bytemask(predictions)
    return bytemask

seeds = {}



def is_useful(seed_id, mutation):
    if random.random() > 0.8:
        return True

    # Iterate diff & bytemask with bitwise AND to determine if bit is useful
    seed = pad_vectorize(open('out/default/queue/' + [x for x in os.listdir('out/default/queue') if 'id:' + seed_id in x][0], 'rb').read())
    mutation = pad_vectorize(mutation)

    difference = diff(seed, mutation).astype('int')
    if seed_id in seeds.keys():
        bytemask = seeds[seed_id]
    else:
        bytemask = query_model(seed.reshape((1,) + seed.shape)).astype('int')
        seeds[seed_id] = bytemask

    if difference.size == bytemask.size and np.any(np.bitwise_and(difference, bytemask)):
        return True
    return False

def vectorize(seed):
    shape = np.zeros(shape=(max_len, bytesize))
    for byte_pos, byte in enumerate(seed):
        bits = bin(byte)[2:].zfill(8)
        for n, bit in enumerate(bits):
            if bit == '1':
                shape[byte_pos, n] = 1.
    return shape.reshape((shape.shape[0]//8,bytesize*8))

def pad(seed):
    if len(seed) < max_len:
        seed = seed + b'\x00' * (max_len - len(seed))
    if len(seed) > max_len:
        seed = seed[:max_len]
    return seed

def pad_vectorize(bytestream):
    return vectorize(pad(bytestream))


def diff(seed, mutation):
    return np.logical_xor(seed, mutation)





def get_bytemask(predictions):
    predictions[predictions < 0] = 0
    predictions[predictions > 0] = 1
    return predictions