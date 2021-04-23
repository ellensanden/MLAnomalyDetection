

def overlap_window (window_size,overlap,seq): # overlap 1 is max. larger number would be less overlap
    import numpy as np
    from numpy import array
    seq = array([seq[i : i + window_size] for i in range(0, len(seq), overlap)]) 
   
    correct = [len(x)==window_size for x in seq]
    seq = seq[correct]
    seq = np.stack(seq, axis=0 )
    seq = seq.reshape(-1,window_size,1)

    return seq



