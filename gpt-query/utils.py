import os


def chunk(batch, mb_size):
    num_batches = (len(batch) + mb_size - 1) // mb_size
    return [batch[i * mb_size : (i+1) * mb_size] for i in range(num_batches)]