# Load session data
import hemiSwap_consts as hconsts
import os
import re
from datetime import datetime
import numpy as np
import numpy.matlib as npm
import pandas as pd
import xarray as xr
import seaborn as sns
import spynal as sp
import matplotlib.pyplot as plt
from spynal.spikes import times_to_bool, rate
from itertools import product
from sklearn.cross_decomposition import CCA
import random
from scipy import stats
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from spynal.matIO import loadmat
from spynal.spikes import psth
import testing as tst
import hemiSwap_consts as hconsts
import json

hemiswap = hconsts.HemiSwap_consts('miller-lab-3', 'tiergan')

def load_and_preprocess_session(session_id,hemiswap,reshape_method):
    '''
    Load and preprocess session data, uses random subsampling of no-swap trials to match swap trials by default
    
    '''

    filepath = os.path.join(hemiswap.loadDir, f'{session_id}.mat')
    ain, ainSchema, analogChnlInfo, electrodeInfo, eventSchema, fileInfo, lfp, lfpSchema,\
    sessionInfo, spikeChnlInfo, spikeTimes, spikeTimesSchema, trialInfo, unitInfo = loadmat(filepath, variables = ['ain', 'ainSchema', 'analogChnlInfo', 'electrodeInfo', 'eventSchema', 'fileInfo', 'lfp', 'lfpSchema',\
    'sessionInfo', 'spikeChnlInfo', 'spikeTimes', 'spikeTimesSchema', 'trialInfo', 'unitInfo'], typemap = {'trialInfo':'DataFrame'}, verbose=True)

    #no swap trials from each side
    noswap_trials_right = np.where((trialInfo['sampleHemifield']=='right') & trialInfo['isSwap']== False) #sample stars from left hemisphere and remains there
    noswap_trials_left = np.where((trialInfo['sampleHemifield']=='left') & trialInfo['isSwap']== False)#sample stars from right hemisphere and remains there

    swap_trials_right = np.where((trialInfo['sampleHemifield']=='right') & trialInfo['isSwap']== True) #sample stars from left hemisphere and remains there
    swap_trials_left = np.where((trialInfo['sampleHemifield']=='left') & trialInfo['isSwap']== True)#sample stars from right hemisphere and remains there
    #noswap_between = hemiswap.poolChannelPairCategories('no_swap','between')

    #define right and left hemisphere electrodes
    right_hemi = np.where(unitInfo['hemisphere']=='right')
    left_hemi = np.where(unitInfo['hemisphere']=='left')

    #analyze only left or right electrodes

    # start from right hemi
    no_swap_spikes_right= np.squeeze(spikeTimes[noswap_trials_right,:])
    right_right_hemi_trials = np.squeeze(no_swap_spikes_right[:,right_hemi])
    right_left_hemi_trials = np.squeeze(no_swap_spikes_right[:,left_hemi])


    #start from left hemi
    no_swap_spikes_left= np.squeeze(spikeTimes[noswap_trials_left,:])
    left_left_hemi_trials = np.squeeze(no_swap_spikes_left[:,left_hemi])
    left_right_hemi_trials = np.squeeze(no_swap_spikes_left[:,right_hemi])

    pre_test = [-700e-3,2300e-3] #based on self.eventTimes = {'sampleOn' : {'Fixation':[-700e-3], 'Sample':[0,700e-3],'Swap cue':[1500e-3], 'Test':[2300e-3]}
    if reshape_method == 'trial_averaged':
        spike_rate, rate_bins = rate(spikeTimes, method = 'bin', lims = pre_test, width=100e-3,step=50e-3)
    elif reshape_method == 'trial_concatenated':
        spike_rate, rate_bins = rate(spikeTimes, method = 'bin', lims = pre_test, width=100e-3,step=100e-3)
    else:
        raise ValueError('reshape_method must be either trial_averaged or trial_concatenated')
    rate_bin_centers = np.mean(rate_bins, axis = 1) #why pre-test? see calculating residuals

    #square root transformation for Gaussian distribution
    spike_rate = np.sqrt(spike_rate)

    #filter out the non-zero neurons

    # their average rate (over time and over trials) needs to be >0.1
    mean_spike_rate = np.mean(spike_rate, axis=(0,2),keepdims=True)
    useful_neurons = np.where(mean_spike_rate>0.1)[1]

    #define right and left hemisphere electrodes
    right_hemi = np.where(unitInfo['hemisphere']=='right')
    left_hemi = np.where(unitInfo['hemisphere']=='left')
    #print(left_hemi[0], right_hemi[0])
    right_hemi = np.intersect1d(right_hemi,useful_neurons)
    left_hemi = np.intersect1d(left_hemi,useful_neurons)

    spike_rate_right = spike_rate[:, right_hemi,:]
    spike_rate_left = spike_rate[:, left_hemi,:]



    spike_rate = spike_rate[:, useful_neurons, :]

    #we need to normalize before calculating the residuals
    #1. Mean pool spike rates across the 200ms before sample object onset
    before_sample_on = [-0.2,0]
    entire_trial = [-1.75,4.5]
    target_bins = np.where((rate_bins >= -0.2) & (rate_bins <= 0))[0]
    # index the bso: dont calculate rate again
    spike_rate_bso = spike_rate[:,:,target_bins] #bso: before sample onset

    spike_rate_bso = np.sqrt(spike_rate_bso)
    spike_rate_bso = np.mean(spike_rate_bso, axis=2, keepdims=True) 
    avg_spike_rate_bso = np.mean(spike_rate_bso, axis=0, keepdims=True)
    std_spike_rate_bso = np.std(spike_rate_bso, axis=0, keepdims=True)
    spike_rate_z_scored = (spike_rate - avg_spike_rate_bso) / std_spike_rate_bso

    #normalize spike_rate right and left (lack of a faster/better way right now)

    #right
    spike_rate_right_bso = spike_rate_right[:,:,target_bins] #bso: before sample onset

    spike_rate_right_bso = np.sqrt(spike_rate_right_bso)
    spike_rate_right_bso = np.mean(spike_rate_right_bso, axis=2, keepdims=True) 
    avg_spike_rate_right_bso = np.mean(spike_rate_right_bso, axis=0, keepdims=True)
    std_spike_rate_right_bso = np.std(spike_rate_right_bso, axis=0, keepdims=True)
    spike_rate_right_z_scored = (spike_rate_right - avg_spike_rate_right_bso) / std_spike_rate_right_bso

    #left
    spike_rate_left_bso = spike_rate_left[:,:,target_bins] #bso: before sample onset

    spike_rate_left_bso = np.sqrt(spike_rate_left_bso)
    spike_rate_left_bso = np.mean(spike_rate_left_bso, axis=2, keepdims=True) 
    avg_spike_rate_left_bso = np.mean(spike_rate_left_bso, axis=0, keepdims=True)
    std_spike_rate_left_bso = np.std(spike_rate_left_bso, axis=0, keepdims=True)
    spike_rate_left_z_scored = (spike_rate_left - avg_spike_rate_left_bso) / std_spike_rate_left_bso


    spike_rate_residuals = spike_rate_z_scored
    spike_rate_residuals_right = spike_rate_right_z_scored
    spike_rate_residuals_left = spike_rate_left_z_scored
    for loc, obj, sh,th in product(set(trialInfo['sampleLoc']),set(trialInfo['sampleObj']),set(trialInfo['sampleHemifield']),set(trialInfo['testHemifield'])):
        condition_trials = np.where((trialInfo['sampleLoc']==loc) & (trialInfo['sampleObj']==obj) 
                                    & (trialInfo['sampleHemifield'] == sh) & (trialInfo['testHemifield']==th))
        print(condition_trials)
        spike_rate_cond = spike_rate_z_scored[condition_trials[0],:,:]
        #print(spike_rate_cond.shape, spike_rate_cond[np.nonzero(spike_rate_cond)])
        
        avg_spike_rate_cond = np.mean(spike_rate_cond,axis=0)
        print(avg_spike_rate_cond)
        spike_rate_residuals[condition_trials[0],:,:] = spike_rate_z_scored[condition_trials[0],:,:] - avg_spike_rate_cond
        print(spike_rate_residuals.shape)

        #right residuals
        spike_rate_cond_right = spike_rate_right_z_scored[condition_trials[0],:,:]
        #print(spike_rate_cond.shape, spike_rate_cond[np.nonzero(spike_rate_cond)])
        
        avg_spike_rate_cond_right = np.mean(spike_rate_cond_right,axis=0)
        print(avg_spike_rate_cond_right)
        spike_rate_residuals_right[condition_trials[0],:,:] = spike_rate_right_z_scored[condition_trials[0],:,:] - avg_spike_rate_cond_right
        print(spike_rate_residuals_right.shape)


        #left residuals
        spike_rate_cond_left = spike_rate_left_z_scored[condition_trials[0],:,:]
        #print(spike_rate_cond.shape, spike_rate_cond[np.nonzero(spike_rate_cond)])
        
        avg_spike_rate_cond_left = np.mean(spike_rate_cond_left,axis=0)
        print(avg_spike_rate_cond_left)
        spike_rate_residuals_left[condition_trials[0],:,:] = spike_rate_left_z_scored[condition_trials[0],:,:] - avg_spike_rate_cond_left
        print(spike_rate_residuals_left.shape)

    noswap_trials_right = np.where((trialInfo['sampleHemifield']=='right') & trialInfo['isSwap']== False) #sample stars from left hemisphere and remains there
    noswap_trials_left = np.where((trialInfo['sampleHemifield']=='left') & trialInfo['isSwap']== False)#sample stars from right hemisphere and remains there
    #noswap_between = hemiswap.poolChannelPairCategories('no_swap','between')

    #swap trials
    swap_trials_right = np.where((trialInfo['sampleHemifield']=='right') & trialInfo['isSwap']== True) #sample stars from left hemisphere and remains there
    swap_trials_left = np.where((trialInfo['sampleHemifield']=='left') & trialInfo['isSwap']== True)#sample stars from right hemisphere and remains there

    right_right_residuals_swap = np.squeeze(spike_rate_residuals_right[swap_trials_right,:,:])
    right_left_residuals_swap = np.squeeze(spike_rate_residuals_left[swap_trials_right,:,:])

    left_left_residuals_swap = np.squeeze(spike_rate_residuals_left[swap_trials_left,:,:])
    left_right_residuals_swap = np.squeeze(spike_rate_residuals_right[swap_trials_left,:,:])


    # keep in mind: we have way less succesful swap trials to work with

    #CHOICE 2: using random subset of no-swap trials, the subset is the same length as the swap trials

    noswap_trials_right_subset = np.array(random.sample(list(noswap_trials_right[0]),len(swap_trials_right[0])))
    noswap_trials_left_subset = np.array(random.sample(list(noswap_trials_left[0]),len(swap_trials_left[0])))
    #print(noswap_trials_right_subset,noswap_trials_left_subset.shape)

    right_right_residuals_sub = np.squeeze(spike_rate_residuals_right[noswap_trials_right_subset,:,:])
    right_left_residuals_sub = np.squeeze(spike_rate_residuals_left[noswap_trials_right_subset,:,:])

    left_left_residuals_sub = np.squeeze(spike_rate_residuals_left[noswap_trials_left_subset,:,:])
    left_right_residuals_sub = np.squeeze(spike_rate_residuals_right[noswap_trials_left_subset,:,:])

    # Print shapes for verification
    
    residuals_dict = {'right-right-no-swap/right-left-no-swap': (right_right_residuals_sub, right_left_residuals_sub),
                  'right-right-swap/right-left-swap':(right_right_residuals_swap,right_left_residuals_swap),
                  'left-right-no-swap/left-left-no-swap': (left_right_residuals_sub, left_left_residuals_sub),
                  'left-right-swap/left-left-swap': (left_right_residuals_swap,left_left_residuals_swap) }
    residuals_list = [(right_right_residuals_sub, right_left_residuals_sub), (right_right_residuals_swap,right_left_residuals_swap),
                  (left_right_residuals_sub, left_left_residuals_sub),(left_right_residuals_swap,left_left_residuals_swap)]
    
    return residuals_dict, residuals_list, rate_bin_centers


def trial_averaged_residuals(X, Y, start, end, rate_bin_centers):
    """
    Calculate trial-averaged residuals for each time period.
    
    Parameters:
    -----------
    X : np.ndarray
        The X residuals
    Y : np.ndarray
        The Y residuals
    time_periods : dict
        Dictionary with time period names as keys and start and end times as values
    rate_bin_centers : np.ndarray
        The rate bin centers
        
    Returns:
    --------
    X_reshaped : np.ndarray(n)
    """
    
    target_bins = np.where((rate_bin_centers >= start) & (rate_bin_centers <= end))[0]
    X_time_win = X[:, :, target_bins]
    Y_time_win = Y[:, :, target_bins]
    
    X_reshaped = np.mean(X_time_win, axis=2)
    Y_reshaped = np.mean(Y_time_win, axis=2)
        
        
    
    return X_reshaped, Y_reshaped

def trial_concatenated_residuals(X, Y):
    """
    Calculate trial-concatenated residuals .
    
    Parameters:
    -----------
    X : np.ndarray
        The X residuals
    Y : np.ndarray
        The Y residuals
   
    Returns:
    --------
    tuple
        The reshaped X and Y residuals
    """
    
    n_trials,n_neurons_right,n_timepoints = X.shape
    n_trials,n_neurons_left, n_timepoints = Y.shape
    X_reshaped = X.reshape(-1,n_neurons_right)
    Y_reshaped = Y.reshape(-1,n_neurons_left)


    
    return X_reshaped, Y_reshaped
def load_results(session_id, output_dir):
    """
    Load testing framework results for a given session.
    
    Parameters:
    -----------
    session_id : str
        The session ID
    output_dir : str
        Base output directory path
        
    Returns:
    --------
    dict
        The loaded results dictionary
    """
    results_dir = os.path.join(output_dir, f'session_{session_id}')
    results_file = os.path.join(results_dir, f'{session_id}.npz')
    
    # Load the compressed npz file
    loaded_data = np.load(results_file, allow_pickle=True)
    
    # Extract the results dictionary
    results = loaded_data['results'].item()
    
    return results
def create_condition_results(residual_results, n_folds='folds_2'):
    """
    Functions that creates dicts where instead of having each residual pair condition combination(4)
    as a seperate dict, we have dicts for each residual pair(2) with the condition as a key.
    """
    condition_results = {}
    for residual_pair in residual_results.keys():
        base_pair = residual_pair.replace('-no-swap', '').replace('-swap', '')
        condition = 'no-swap' if 'no-swap' in residual_pair else 'swap'
        
        if base_pair not in condition_results:
            condition_results[base_pair] = {}
        
        condition_results[base_pair][condition] = residual_results[residual_pair][n_folds]
    
    return condition_results

######## - ACROSS SESSIONS - #########

def load_multi_session_results(results_dir,subject_id,hemiswap=hemiswap):
    """
    Load results from a multi-session analysis.
    
    Parameters:
    -----------
    results_dir : str
        Path to the main results directory
    
    Returns:
    --------
    dict
        Analysis metadata
    dict
        Results for all sessions
    """
    # Load metadata
    with open(os.path.join(results_dir, 'analysis_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load results for each session
    all_sessions_results = {}
    sessions = [session_id for session_id in hemiswap.sessions['full'] if f'{subject_id}' in session_id]
    for idx in metadata['parameters']['selected_sessions']:
        session_id = sessions[idx]
        session_file = os.path.join(results_dir, f'session_{session_id}', 
                                  f'session_{session_id}_results.npz')
        try:
            loaded_data = np.load(session_file, allow_pickle=True)
            all_sessions_results[session_id] = loaded_data['results'].item()
        except Exception as e:
            print(f"Error loading session {session_id}: {str(e)}")
            continue
    
    return metadata, all_sessions_results

def mean_std_across_sessions(results_dir, subject_id, cv):
    metadata, all_sessions_results = load_multi_session_results(results_dir, subject_id)
    
    # Get the time periods from metadata
    time_periods = list(metadata['parameters']['time_periods'].keys())
    residual_pairs = list(next(iter(all_sessions_results.values())).keys())
    
    # Initialize dictionaries to store correlations for each residual pair and time period
    corrs_across_sessions = {rp: {tp: [] for tp in time_periods} for rp in residual_pairs}
    
    # Collect correlations across sessions
    for session_results in all_sessions_results.values():
        for residual_pair in residual_pairs:
            for time_period in time_periods:
                # Get the first dimension correlation and square it
                correlation = session_results[residual_pair][cv][time_period]['correlations'][0]**2
                corrs_across_sessions[residual_pair][time_period].append(correlation)
    
    # Calculate mean and std across sessions
    mean_corrs_folds_sessions = {}
    std_corrs_folds_sessions = {}
    
    for residual_pair in residual_pairs:
        mean_corrs_folds_sessions[residual_pair] = {}
        std_corrs_folds_sessions[residual_pair] = {}
        
        for time_period in time_periods:
            correlations = np.array(corrs_across_sessions[residual_pair][time_period])
            mean_corrs_folds_sessions[residual_pair][time_period] = np.mean(correlations)
            std_corrs_folds_sessions[residual_pair][time_period] = np.std(correlations)
    
    return mean_corrs_folds_sessions, std_corrs_folds_sessions, time_periods, residual_pairs
