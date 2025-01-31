import hemiSwap_consts as hconsts
from load_sessions import load_neural_data
import numpy as np
from spynal.spikes import rate
from itertools import product
import random
import h5py
import numpy as np
import os
from datetime import datetime


import numpy as np
from itertools import product
import random

def calculate_spike_rates(spike_times, pre_test=(-700e-3, 2300e-3), bin_width=100e-3, bin_step=50e-3):
    """Calculate spike rates with given parameters."""
    spike_rate, rate_bins = rate(spike_times, method='bin', lims=pre_test, 
                                width=bin_width, step=bin_step)
    rate_bin_centers = np.mean(rate_bins, axis=1)
    return np.sqrt(spike_rate), rate_bins, rate_bin_centers

def filter_neurons(spike_rate, unit_info):
    """Filter neurons based on mean spike rate and hemisphere."""
    mean_spike_rate = np.mean(spike_rate, axis=(0,2), keepdims=True)
    useful_neurons = np.where(mean_spike_rate > 0.1)[1]
    
    hemispheres = {
        'right': np.intersect1d(np.where(unit_info['hemisphere'] == 'right'), useful_neurons),
        'left': np.intersect1d(np.where(unit_info['hemisphere'] == 'left'), useful_neurons)
    }
    
    return useful_neurons, hemispheres

def normalize_spike_rates(spike_rate, target_bins):
    """Z-score normalize spike rates based on baseline period."""
    spike_rate_bso = spike_rate[:, :, target_bins]
    spike_rate_bso = np.sqrt(spike_rate_bso)
    spike_rate_bso = np.mean(spike_rate_bso, axis=2, keepdims=True)
    
    avg_spike_rate_bso = np.mean(spike_rate_bso, axis=0, keepdims=True)
    std_spike_rate_bso = np.std(spike_rate_bso, axis=0, keepdims=True)
    
    return (spike_rate - avg_spike_rate_bso) / std_spike_rate_bso

def calculate_residuals(spike_rate_z_scored, trial_info):
    """Calculate residuals for given spike rates based on trial conditions."""
    spike_rate_residuals = spike_rate_z_scored.copy()
    
    conditions = product(
        set(trial_info['sampleLoc']),
        set(trial_info['sampleObj']),
        set(trial_info['sampleHemifield']),
        set(trial_info['testHemifield'])
    )
    
    for loc, obj, sh, th in conditions:
        condition_trials = np.where(
            (trial_info['sampleLoc'] == loc) & 
            (trial_info['sampleObj'] == obj) & 
            (trial_info['sampleHemifield'] == sh) & 
            (trial_info['testHemifield'] == th)
        )[0]
        
        spike_rate_cond = spike_rate_z_scored[condition_trials, :, :]
        avg_spike_rate_cond = np.mean(spike_rate_cond, axis=0)
        spike_rate_residuals[condition_trials, :, :] -= avg_spike_rate_cond
    
    return spike_rate_residuals



####### alternative process session function

def get_trial_residuals(residuals, trial_info, hemi, is_swap):
    """
    Extract residuals for specific trial conditions with proper handling of sample sizes.
    
    Parameters:
    -----------
    residuals : np.ndarray
        The residuals array to sample from
    trial_info : dict
        Trial information containing condition data
    hemi : str
        Hemisphere ('right' or 'left')
    is_swap : bool
        Whether to get swap or no-swap trials
    
    Returns:
    --------
    tuple : (residuals_array, trial_indices)
    """
    # Get initial trial indices
    trials = np.where(
        (trial_info['sampleHemifield'] == hemi) & 
        (trial_info['isSwap'] == is_swap)
    )[0]
    
    if not is_swap:
        # Get number of swap trials for this hemisphere
        swap_trials = np.where(
            (trial_info['sampleHemifield'] == hemi) & 
            (trial_info['isSwap'] == True)
        )[0]
        n_swap = len(swap_trials)
        
        # If we need to sample no-swap trials
        if n_swap > 0:
            # Handle case where we have fewer no-swap trials than swap trials
            if len(trials) < n_swap:
                print(f"Warning: {hemi} hemisphere has fewer no-swap trials ({len(trials)}) "
                      f"than swap trials ({n_swap}). Using all available no-swap trials.")
                n_samples = len(trials)
            else:
                n_samples = n_swap
                
            # Sample trials
            if len(trials) > 1:  # Only sample if we have more than one trial
                trials = np.array(random.sample(list(trials), n_samples))
    
    # Extract residuals
    trial_residuals = np.squeeze(residuals[trials, :, :])
    
    return trial_residuals, trials

def process_session(session_data):
    """
    Process a single session's data with improved error handling.
    
    Parameters:
    -----------
    session_data : dict
        Dictionary containing session data
    
    Returns:
    --------
    dict : Processed residuals for different conditions
    """
    try:
        # Extract required data
        spike_times = session_data['spikeTimes']
        trial_info = session_data['trialInfo']
        unit_info = session_data['unitInfo']
        
        # Calculate spike rates
        spike_rate, rate_bins, _ = calculate_spike_rates(spike_times)
        
        # Filter neurons and get hemispheres
        useful_neurons, hemispheres = filter_neurons(spike_rate, unit_info)
        
        # Prepare hemisphere-specific spike rates
        spike_rates = {
            'all': spike_rate[:, useful_neurons, :],
            'right': spike_rate[:, hemispheres['right'], :],
            'left': spike_rate[:, hemispheres['left'], :]
        }
        
        # Get baseline period bins
        target_bins = np.where((rate_bins >= -0.2) & (rate_bins <= 0))[0]
        
        # Normalize spike rates
        spike_rates_z = {
            hemi: normalize_spike_rates(rates, target_bins)
            for hemi, rates in spike_rates.items()
        }
        
        # Calculate residuals
        residuals = {
            hemi: calculate_residuals(rates, trial_info)
            for hemi, rates in spike_rates_z.items()
        }
        
        # Process results for each hemisphere and condition
        results = {}
        for hemi in ['right', 'left']:
            for is_swap in [False, True]:
                suffix = '_swap' if is_swap else '_sub'
                
                try:
                    # Get residuals for both hemispheres for these trials
                    trials_residuals, _ = get_trial_residuals(
                        residuals['all'], trial_info, hemi, is_swap
                    )
                    
                    # Store results
                    results[f'{hemi}_{hemi}_residuals{suffix}'] = trials_residuals
                    other_hemi = 'left' if hemi == 'right' else 'right'
                    results[f'{hemi}_{other_hemi}_residuals{suffix}'] = trials_residuals
                    
                except Exception as e:
                    print(f"Warning: Error processing {hemi} hemisphere, {is_swap} condition: {str(e)}")
                    # Initialize with empty arrays or appropriate placeholder
                    results[f'{hemi}_{hemi}_residuals{suffix}'] = np.array([])
                    results[f'{hemi}_{other_hemi}_residuals{suffix}'] = np.array([])
        
        return results
    
    except Exception as e:
        print(f"Error in process_session: {str(e)}")
        raise

# Main processing loop with error handling
def process_all_sessions(sessions):
    """
    Process all sessions with proper error handling.
    
    Parameters:
    -----------
    sessions : dict
        Dictionary of session data
    
    Returns:
    --------
    dict : Processed residuals for all sessions
    """
    residuals_sessions = {}
    
    for session_id in sessions:
        try:
            print(f"Processing session {session_id}")
            residuals_sessions[session_id] = process_session(sessions[session_id])
            print(f"Successfully processed session {session_id}")
        except Exception as e:
            print(f"Error processing session {session_id}: {str(e)}")
            continue
    
    return residuals_sessions

####### alternative process all sessions function

def save_to_hdf5(residuals_sessions, filepath):
    """
    Save residuals data to HDF5 format with compression.
    
    Parameters:
    -----------
    residuals_sessions : dict
        Dictionary containing residuals data for each session
    filepath : str
        Path where the HDF5 file should be saved
    """
    try:
        with h5py.File(filepath, 'w') as f:
            # Add metadata
            f.attrs['creation_date'] = str(datetime.now())
            f.attrs['data_type'] = 'neural_residuals'
            f.attrs['num_sessions'] = len(residuals_sessions)
            
            # Create a group for each session
            for session_id, session_data in residuals_sessions.items():
                session_group = f.create_group(str(session_id))
                
                # Save each type of residual
                for residual_type, data in session_data.items():
                    # Use compression for efficient storage
                    session_group.create_dataset(
                        residual_type,
                        data=data,
                        compression='gzip',
                        compression_opts=9  # Maximum compression
                    )
        return True
    except Exception as e:
        print(f"Error saving HDF5 file: {e}")
        return False

def save_to_npz(residuals_sessions, filepath):
    """
    Save residuals data to compressed NPZ format.
    
    Parameters:
    -----------
    residuals_sessions : dict
        Dictionary containing residuals data for each session
    filepath : str
        Path where the NPZ file should be saved
    """
    try:
        # Flatten the dictionary structure for npz format
        save_dict = {}
        for session_id, session_data in residuals_sessions.items():
            for residual_type, data in session_data.items():
                # Create unique keys for each session and residual type
                key = f"{session_id}_{residual_type}"
                save_dict[key] = data
                
        # Save to compressed npz
        np.savez_compressed(filepath, **save_dict)
        return True
    except Exception as e:
        print(f"Error saving NPZ file: {e}")
        return False

def load_from_hdf5(filepath):
    """
    Load residuals data from HDF5 file.
    
    Parameters:
    -----------
    filepath : str
        Path to the HDF5 file
    
    Returns:
    --------
    dict
        Dictionary containing the loaded residuals data
    """
    try:
        residuals_sessions = {}
        with h5py.File(filepath, 'r') as f:
            # Print metadata
            print(f"File created on: {f.attrs['creation_date']}")
            print(f"Data type: {f.attrs['data_type']}")
            
            # Load each session
            for session_id in f.keys():
                residuals_sessions[session_id] = {}
                for residual_type in f[session_id].keys():
                    residuals_sessions[session_id][residual_type] = f[session_id][residual_type][:]
        
        return residuals_sessions
    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        return None

def load_from_npz(filepath):
    """
    Load residuals data from NPZ file.
    
    Parameters:
    -----------
    filepath : str
        Path to the NPZ file
    
    Returns:
    --------
    dict
        Dictionary containing the loaded residuals data
    """
    try:
        # Load the npz file
        data = np.load(filepath)
        
        # Reconstruct the nested dictionary structure
        residuals_sessions = {}
        for key in data.files:
            session_id, residual_type = key.split('_', 1)
            if session_id not in residuals_sessions:
                residuals_sessions[session_id] = {}
            residuals_sessions[session_id][residual_type] = data[key]
            
        return residuals_sessions
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return None


def save_results(residuals_sessions, base_path, session_number, subject):
    """
    Save results in both formats with appropriate file naming.
    
    Parameters:
    -----------
    residuals_sessions : dict
        Dictionary containing residuals data
    base_path : str
        Base directory for saving files
    session_date : str
        Date identifier for the session
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hdf5_path = os.path.join(base_path, f"{subject}_residuals_{session_number}_{timestamp}.h5")
    npz_path = os.path.join(base_path, f"{subject}_residuals_{session_number}_{timestamp}.npz")
    
    # Save in both formats
    hdf5_success = save_to_hdf5(residuals_sessions, hdf5_path)
    npz_success = save_to_npz(residuals_sessions, npz_path)
    
    # Report results
    if hdf5_success:
        print(f"Successfully saved HDF5 file: {hdf5_path}")
    if npz_success:
        print(f"Successfully saved NPZ file: {npz_path}")