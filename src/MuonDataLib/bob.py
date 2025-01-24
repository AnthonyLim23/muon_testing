import h5py
import numpy as np
import time

def load_data_dict(file_name):
    start_timer = time.time()

    file = h5py.File(file_name, 'r')
    tmp = file.require_group('raw_data_1')
    tmp = tmp.require_group('detector_1')

    N = tmp['event_id'].len()

    IDs = np.zeros(N, dtype=int)
    times = np.zeros(N, dtype=np.double)
    amps = np.zeros(N, dtype=np.double)
    
    M = tmp['event_index'].len()
    start_j = np.zeros(M, dtype=int)
    start_t = np.zeros(M, dtype=np.double)
    
    data = {'event_id': IDs,
            'event_time_offset': times,
            'pulse_height': amps,
            'event_index': start_j,
            'event_time_zero': start_t}
    N_data = len(data)
    for key in data.keys():
        tmp[key].read_direct(data[key])

    #periods = np.asarray(tmp['period_number'][:])
    file.close()
    
    return time.time() - start_timer, IDs, start_j, times, amps, start_t

