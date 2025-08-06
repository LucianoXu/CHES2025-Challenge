import math
import random
from typing import Literal

import h5py
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn.functional as F

from numba import njit

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])
AES_Sbox_inv =  np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

def HW(s):
    '''
    Calculate the Hamming Weight of a byte.
    '''
    return bin(s).count("1")

def calculate_HW(data: list[int]|np.ndarray) -> np.ndarray:
    '''
    Calculate the Hamming Weight of a list of bytes.
    '''
    hw = [bin(x).count("1") for x in range(256)]
    return np.array([hw[int(s)] for s in data])


def load_ctf_2025(
        filename: str,
        byte: int = 0, 
        train_begin: int = 0, train_end: int = 100000, test_begin: int = 0, test_end: int = 50000):
    '''
    filename: path to the h5 dataset
    leakage_model: the leakage model we consider. 'HW' for Hamming Weight, 'ID' for Intermediate Value.
    byte: the byte we consider, 0 for the first byte.

    Returns:
        (X_profiling, X_attack), (Y_profiling, Y_attack), (P_profiling, P_attack), (K_profiling, K_attack)
        where X is the traces, Y is the labels (byte after AES_Sbox operation), P is the plaintexts, and K is the keys.
    '''

    in_file = h5py.File(filename, "r")

    # get the traces
    X_profiling = np.array(in_file['Profiling_traces/traces']) # (num_example, dim) : (500_000, 7_000)
    assert X_profiling.ndim == 2, "Profiling traces should be 2D array."

    # get the plaintexts
    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, byte])   # type: ignore # (num_example,) : (500_000,)

    K_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'][:, byte])   # type: ignore # (num_example,) : (500_000,)
    # by comparing this two branches, we observe that the labels are bytes after the AES_Sbox operation.
    if byte != 0:
        Y_profiling = np.zeros(P_profiling.shape[0])
        print("Loading Y_profiling")
        for i in range(len(P_profiling)): #tqdm()
            Y_profiling[i] = AES_Sbox[P_profiling[i] ^ K_profiling[i]]
    else:
        # labels are for byte 0
        Y_profiling = np.array(in_file['Profiling_traces/metadata'][:]['labels'])   # type: ignore # (num_example,) : (500_000,)

    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'])    # (num_example, dim) : (100_000, 7_000)
    assert X_attack.ndim == 2, "Attack traces should be 2D array."

    P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, byte]) # type: ignore # (num_example,) : (100_000,)
    K_attack = np.array(in_file['Attack_traces/metadata'][:]['key'][:, byte])   # type: ignore # (num_example,) : (100_000,)

    if byte != 0:
        print("Loading Y_attack")
        Y_attack = np.zeros(P_attack.shape[0])
        for i in range(len(P_attack)):
            Y_attack[i] = AES_Sbox[P_attack[i] ^ K_attack[i]]

    else:
        Y_attack = np.array(in_file['Attack_traces/metadata'][:]['labels'])  # type: ignore # (num_example,) : (100_000,)

    print("Information about the dataset: ")
    print("X_profiling total shape", X_profiling.shape)
    print("Y_profiling total shape", Y_profiling.shape)
    print("P_profiling total shape", P_profiling.shape)

    print("X_attack total shape", X_attack.shape)
    print("Y_attack total shape", Y_attack.shape)
    print("P_attack total shape", P_attack.shape)

    # we know that the key is the same for all traces in the attack split
    print("correct key:", K_attack[0])
    print()


    return  (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), \
            (Y_profiling[train_begin:train_end], Y_attack[test_begin:test_end]), \
            (P_profiling[train_begin:train_end], P_attack[test_begin:test_end]), \
            (K_profiling[train_begin:train_end], K_attack[test_begin:test_end])


# Objective: GE
def rk_key(rank_array: np.ndarray, key):
    '''
    Calculate the rank of the key in the rank_array.

    rank_array: array of ranks for each key, therefore the shape is (256,).
    '''

    key_val = rank_array[key]
    final_rank = np.float32(np.where(np.sort(rank_array)[::-1] == key_val)[0][0])

    if math.isnan(float(final_rank)) or math.isinf(float(final_rank)):
        return np.float32(256)
    else:
        return np.float32(final_rank)


# Hamming weight lookup table for njit optimization
HW_LOOKUP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)

@njit
def rank_compute_hw_njit(prediction, att_plt, correct_key, sbox):
    '''
    Optimized version for Hamming Weight leakage model using njit.
    '''
    (nb_traces, nb_hyp) = prediction.shape
    
    key_log_prob = np.zeros(256, dtype=np.float64)
    prediction_log = np.log(prediction + 1e-40)
    rank_evol = np.full(nb_traces, 255, dtype=np.float32)
    
    for i in range(nb_traces):
        for k in range(256):
            intermediate = sbox[k ^ att_plt[i]]
            hw_value = HW_LOOKUP[intermediate]
            key_log_prob[k] += prediction_log[i, hw_value]
            
        # Inline rk_key functionality
        key_val = key_log_prob[correct_key]
        sorted_probs = np.sort(key_log_prob)[::-1]
        rank = np.where(sorted_probs == key_val)[0][0]
        rank_evol[i] = np.float32(rank)

    return rank_evol, key_log_prob

@njit
def rank_compute_id_njit(prediction, att_plt, correct_key, sbox):
    '''
    Optimized version for Intermediate Value leakage model using njit.
    '''
    (nb_traces, nb_hyp) = prediction.shape
    
    key_log_prob = np.zeros(256, dtype=np.float64)
    prediction_log = np.log(prediction + 1e-40)
    rank_evol = np.full(nb_traces, 255, dtype=np.float32)
    
    for i in range(nb_traces):
        for k in range(256):
            y_value = sbox[k ^ att_plt[i]]
            key_log_prob[k] += prediction_log[i, y_value]
            
        # Inline rk_key functionality
        key_val = key_log_prob[correct_key]
        sorted_probs = np.sort(key_log_prob)[::-1]
        rank = np.where(sorted_probs == key_val)[0][0]
        rank_evol[i] = np.float32(rank)

    return rank_evol, key_log_prob

# Compute the evolution of rank
def rank_compute(prediction, att_plt, correct_key, leakage_fn):
    '''
    :param prediction: prediction by the neural network (probability)
    :param att_plt: attack plaintext
    :return: key_log_prob which is the log probability
    '''
    
    (nb_traces, nb_hyp) = prediction.shape

    # note that the key_log_prob is accumulated, therefore this reflects the evolution of the rank.
    key_log_prob = np.zeros(256)
    prediction = np.log(prediction + 1e-40)
    rank_evol = np.full(nb_traces, 255)
    for i in range(nb_traces):
        for k in range(256):
            y_value = leakage_fn(att_plt[i], k)
            key_log_prob[k] += prediction[i, y_value]
            
        rank_evol[i] = rk_key(key_log_prob, correct_key) # rk_key will sort key_log_prob.

    return rank_evol, key_log_prob

def rank_compute_optimized(prediction, att_plt, correct_key, leakage_model='HW'):
    '''
    Optimized version that automatically chooses the appropriate njit function.
    
    :param prediction: prediction by the neural network (probability)
    :param att_plt: attack plaintext
    :param correct_key: the correct key byte
    :param leakage_model: 'HW' for Hamming Weight, 'ID' for Intermediate Value
    :return: rank_evol, key_log_prob
    '''
    att_plt = att_plt.astype(np.int32)
    
    if leakage_model == 'HW':
        return rank_compute_hw_njit(prediction, att_plt, correct_key, AES_Sbox)
    elif leakage_model == 'ID':
        return rank_compute_id_njit(prediction, att_plt, correct_key, AES_Sbox)
    else:
        raise ValueError(f"Unsupported leakage model: {leakage_model}")


def perform_attacks(nb_traces: int, predictions: np.ndarray, plt_attack, correct_key, leakage_fn, nb_attacks=1, shuffle=True):
    '''
    :param nb_traces: number_traces used to attack
    :param predictions: output of the neural network i.e. prob of each class
    :param plt_attack: plaintext from attack traces
    :param nb_attacks: number of attack experiments
    :param byte: byte in questions
    :param shuffle: true then it shuffle
    :return: mean of the rank for each experiments, log_probability of the output for all key
    '''

    # prediction: ()

    all_rk_evol = np.zeros((nb_attacks, nb_traces)) #(num_attack, num_traces used)
    all_key_log_prob = np.zeros(256)
    for i in tqdm(range(nb_attacks)): #tqdm()
        if shuffle:
            l = list(zip(predictions, plt_attack)) #list of [prediction, plaintext_attack]
            random.shuffle(l) #shuffle the each other prediction
            sp, splt = list(zip(*l)) #*l = unpacking, output: shuffled predictions and shuffled plaintext.
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces] #just use the required number of traces
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]
        rank_evol, key_log_prob = rank_compute(att_pred, att_plt,correct_key,leakage_fn=leakage_fn)
        all_rk_evol[i] = rank_evol
        all_key_log_prob += key_log_prob

    # return the mean rank evolution across different attacks.
    return np.mean(all_rk_evol, axis=0), key_log_prob, #this will be the last one key_log_prob

def perform_attacks_optimized(nb_traces: int, predictions: np.ndarray, plt_attack, correct_key, leakage_model='HW', nb_attacks=1, shuffle=True):
    '''
    Optimized version using njit functions.
    
    :param nb_traces: number_traces used to attack
    :param predictions: output of the neural network i.e. prob of each class
    :param plt_attack: plaintext from attack traces
    :param correct_key: the correct key byte
    :param leakage_model: 'HW' for Hamming Weight, 'ID' for Intermediate Value
    :param nb_attacks: number of attack experiments
    :param shuffle: true then it shuffle
    :return: mean of the rank for each experiments, log_probability of the output for all key
    '''

    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    all_key_log_prob = np.zeros(256)
    
    for i in tqdm(range(nb_attacks)):
        if shuffle:
            l = list(zip(predictions, plt_attack))
            random.shuffle(l)
            sp, splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]
        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]
            
        rank_evol, key_log_prob = rank_compute_optimized(att_pred, att_plt, correct_key, leakage_model)
        all_rk_evol[i] = rank_evol
        all_key_log_prob += key_log_prob

    return np.mean(all_rk_evol, axis=0), key_log_prob



def proba_to_index( proba, classes):
    number_traces = proba.shape[0]
    prediction = np.zeros((number_traces))
    for i in range(number_traces):
        sorted_index = np.argsort(proba[i])
        # Store the index of the most possible cluster
        prediction[i] = classes[sorted_index[-1]]
    return prediction

def attack_calculate_metrics(model, nb_attacks, nb_traces_attacks,correct_key, X_attack, Y_attack, plt_attack, leakage):
    # Test: Attack on the test traces
    container = np.zeros((1+256+nb_traces_attacks,))
    predictions = model.predict(X_attack[:nb_traces_attacks])
    print("predictions:",predictions.shape)
    if leakage == 'HW':
        classes = 9
    elif leakage == 'ID':
        classes = 256
    classes_labels = range(classes)
    Y_pred =  proba_to_index(predictions, classes_labels)
    accuracy = accuracy_score(Y_attack[:nb_traces_attacks], Y_pred)
    print('accuracy: ', accuracy)
    
    # Use the optimized version
    avg_rank, all_rank = perform_attacks_optimized(nb_traces_attacks, predictions, plt_attack, correct_key, leakage_model=leakage, nb_attacks=nb_attacks, shuffle=True)

    #calculate GE
    container[257:] = avg_rank
    container[1:257] = all_rank

    # calculate accuracy
    container[0] = accuracy
    return container


def NTGE_fn(GE):
    '''
    Find the index that the GE is stabilized to zero after it.
    '''
    NTGE = float('inf')
    for i in range(GE.shape[0] - 1, -1, -1):
        if GE[i] > 0:
            break
        elif GE[i] == 0:
            NTGE = i
    return NTGE


def evaluate(device, model, X_attack, plt_attack,correct_key,leakage_fn, nb_attacks=100, total_nb_traces_attacks=2000, nb_traces_attacks = 1700):
    attack_traces = torch.from_numpy(X_attack[:total_nb_traces_attacks]).to(device).unsqueeze(1).float()
    predictions_wo_softmax = model(attack_traces)
    predictions = F.softmax(predictions_wo_softmax, dim=1)
    predictions = predictions.cpu().detach().numpy()
    GE, key_prob = perform_attacks(nb_traces_attacks, predictions, plt_attack, correct_key,
                                   nb_attacks=nb_attacks, shuffle=True, leakage_fn=leakage_fn)
    NTGE = NTGE_fn(GE)
    print("GE", GE)
    print("NTGE", NTGE)
    return GE,NTGE

def evaluate_optimized(device, model, X_attack, plt_attack, correct_key, leakage_model='HW', nb_attacks=100, total_nb_traces_attacks=2000, nb_traces_attacks=1700):
    """
    Optimized version of evaluate function using njit-optimized rank computation.
    
    :param device: torch device
    :param model: trained model
    :param X_attack: attack traces
    :param plt_attack: attack plaintexts
    :param correct_key: correct key byte
    :param leakage_model: 'HW' for Hamming Weight, 'ID' for Intermediate Value
    :param nb_attacks: number of attack experiments
    :param total_nb_traces_attacks: total number of attack traces to load
    :param nb_traces_attacks: number of traces to use in each attack
    :return: GE, NTGE
    """
    attack_traces = torch.from_numpy(X_attack[:total_nb_traces_attacks]).to(device).unsqueeze(1).float()
    predictions_wo_softmax = model(attack_traces)
    predictions = F.softmax(predictions_wo_softmax, dim=1)
    predictions = predictions.cpu().detach().numpy()
    
    GE, key_prob = perform_attacks_optimized(nb_traces_attacks, predictions, plt_attack, correct_key, 
                                           leakage_model=leakage_model, nb_attacks=nb_attacks, shuffle=True)
    NTGE = NTGE_fn(GE)
    print("GE", GE)
    print("NTGE", NTGE)
    return GE, NTGE