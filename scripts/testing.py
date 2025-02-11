import cca_functions as cca
import plots as plts
import load_session as lps
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import numpy as np
import os
import json
from datetime import datetime
import gc  # For garbage collection
import hemiSwap_consts as hconsts
from contextlib import contextmanager
import psutil
hemiswap = hconsts.HemiSwap_consts('miller-lab-3', 'tiergan')

@contextmanager
def memory_guard():
        """Context manager to ensure memory cleanup"""
        try:
            yield
        finally:
            gc.collect()

def process_single_session(hemiswap, session_id, n_folds_list, time_periods, reg_params, output_dir,reshape_method,br=False):
    """
    Run the testing framework with L2 regularization and save results for each time period.
    """
    with memory_guard():
        residuals_dict, _, rate_bin_centers = lps.load_and_preprocess_session(session_id, hemiswap,reshape_method=reshape_method)
        results_dir = os.path.join(output_dir, f'session_{session_id}/{reshape_method}')
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    for residuals, (X, Y) in zip(residuals_dict.keys(), residuals_dict.values()):
        all_results[residuals] = {}
        residual_X = residuals.split('/')[0]
        residual_Y = residuals.split('/')[1]
        for n_folds in n_folds_list:
            all_results[residuals][f'folds_{n_folds}'] = {}
        with memory_guard():
            for time_period, (start, end) in time_periods.items():
                print(time_period)
                target_bins = np.where((rate_bin_centers >= start) & (rate_bin_centers <= end))[0]
                X_time_win = X[:, :, target_bins]
                Y_time_win = Y[:, :, target_bins]
                if reshape_method == 'trial_averaged':
                    X_reshaped = np.mean(X_time_win, axis=2)
                    Y_reshaped = np.mean(Y_time_win, axis=2)
                elif reshape_method == 'trial_concatenated':
                    X_reshaped, Y_reshaped = lps.trial_concatenated_residuals(X_time_win,Y_time_win)
                else:
                    raise ValueError('reshape_method must be either "trial_averaged" or "trial_concatenated"')
                del X_time_win,Y_time_win
                gc.collect()
                for n_folds in n_folds_list:
                    all_results[residuals][f'folds_{n_folds}'][time_period] = {}
                    title = f'{n_folds}_folds_{residual_X}-{residual_Y}-{time_period}'
                    #kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

                    folds_results = cca.perform_cca(X_reshaped, Y_reshaped, residual_X, residual_Y, title,results_dir,cv=n_folds,reg=True,reg_params=reg_params)
                    
                    #avg_correlations = folds_results['correlations'] #np.mean(correlations_all_folds, axis=0)
                    all_results[residuals][f'folds_{n_folds}'][time_period] = folds_results

            
        
        if br:
            break
    # Save all results to a numpy file
    results_file = os.path.join(results_dir, f'{session_id}_{reshape_method}.npz')
    np.savez_compressed(results_file, results=all_results)
    
    return all_results





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


def get_session_selection(total_sessions, step_size):
    """
    Calculate which sessions to analyze based on total sessions and step size.
    
    Parameters:
    -----------
    total_sessions : int
        Total number of available sessions
    step_size : int
        How many sessions to skip between selections
        
    Returns:
    --------
    list
        Selected session indices
    """
    return list(range(1, total_sessions + 1, step_size))

def save_analysis_metadata(output_dir, analysis_params):
    """
    Save analysis parameters and session selection information.
    """
    metadata_file = os.path.join(output_dir, 'analysis_metadata.json')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        'timestamp': timestamp,
        'parameters': analysis_params
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def process_multi_sessions(hemiswap, subject_id, total_sessions, rate_bin_params, train_test_ratio, reshape_method,
                                 sessions, step_size, n_folds_list, 
                              time_periods, reg_params, output_dir, all_sessions=False,batch_size=1):
    """
    Run the testing framework across multiple sessions with memory management.
    
    Parameters:
    -----------
    hemiswap : bool
        Hemisphere swap parameter
    total_sessions : int
        Total number of available sessions
    step_size : int
        How many sessions to skip between selections
    n_folds_list : list
        List of fold numbers for cross-validation
    time_periods : dict
        Dictionary of time periods to analyze
    reg_params : dict
        Regularization parameters
    output_dir : str
        Output directory path
    batch_size : int, optional
        Number of sessions to process before forcing garbage collection

    ##Note: if reshape_method is missing, then it is 'trial_averaged'
    #added: all_sessions parameter to run all sessions when i need to 
    """
    if all_sessions:
        selected_sesions = list(range(1,total_sessions+1))
    else:
    # Calculate which sessions to analyze
        selected_sessions = get_session_selection(total_sessions, step_size)
    
    # Create main results directory
    main_results_dir = os.path.join(output_dir, f'multi_session_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Save analysis parameters
    analysis_params = {
        'total_sessions': total_sessions,
        'step_size': step_size,
        'selected_sessions': selected_sessions,
        'n_folds_list': n_folds_list,
        'time_periods': {k: list(v) for k, v in time_periods.items()},
        'reg_params': list(reg_params),
        'subject_id': subject_id,
        'rate_bin_params': rate_bin_params,
        'train_test_ratio': train_test_ratio,
        'reshape_method': reshape_method
    }
    save_analysis_metadata(main_results_dir, analysis_params)
    
    # Process sessions in batches
    for batch_start in range(0, len(selected_sessions), batch_size):
        batch_sessions = selected_sessions[batch_start:batch_start + batch_size]
        print(selected_sessions,batch_sessions)
        for idx in batch_sessions:
            session_id = sessions[idx]
            print(f"Processing session {session_id}")
            try:
                # Create session-specific directory
                session_results_dir = os.path.join(main_results_dir, f'session_{session_id}/{reshape_method}')
                os.makedirs(session_results_dir, exist_ok=True)
                
                # Load and process session data
                residuals_dict, _, rate_bin_centers = lps.load_and_preprocess_session(session_id, hemiswap,reshape_method)
                
                all_results = {}
                for residuals, (X, Y) in zip(residuals_dict.keys(), residuals_dict.values()):
                    all_results[residuals] = {}
                    residual_X = residuals.split('/')[0]
                    residual_Y = residuals.split('/')[1]
                    
                    for n_folds in n_folds_list:
                        all_results[residuals][f'folds_{n_folds}'] = {}

                    
                    for time_period, (start, end) in time_periods.items():
                        print(f"  Processing {time_period}")
                        target_bins = np.where((rate_bin_centers >= start) & (rate_bin_centers <= end))[0]
                        X_time_win = X[:, :, target_bins]
                        Y_time_win = Y[:, :, target_bins]
                        if reshape_method == 'trial_averaged':
                            X_reshaped = np.mean(X_time_win, axis=2)
                            Y_reshaped = np.mean(Y_time_win, axis=2)
                        elif reshape_method == 'trial_concatenated':
                            X_reshaped, Y_reshaped = lps.trial_concatenated_residuals(X_time_win,Y_time_win)
                        else:
                            raise ValueError('reshape_method must be either "trial_averaged" or "trial_concatenated"')
                        for n_folds in n_folds_list:
                            title = f'{n_folds}_folds_{residual_X}-{residual_Y}-{time_period}'
                            folds_results = cca.perform_cca(X_reshaped, Y_reshaped, 
                                                        residual_X, residual_Y, 
                                                        title, session_results_dir,
                                                        cv=n_folds, reg=True, 
                                                        reg_params=reg_params)
                            
                            all_results[residuals][f'folds_{n_folds}'][time_period] = folds_results
                    
                
                # Save results for this session
                results_file = os.path.join(session_results_dir, f'session_{session_id}_results_{reshape_method}.npz')
                np.savez_compressed(results_file, results=all_results)
                del all_results
                gc.collect()
                
            except Exception as e:
                print(f"Error processing session {session_id}: {str(e)}")
                continue
                
        # Force garbage collection after each batch
        gc.collect()
    
    return main_results_dir

########### - multisession function but with extra attention on memory usage for when running all sessions - #######
def process_multi_sessions_memory(hemiswap, subject_id, total_sessions, rate_bin_params, train_test_ratio, reshape_method,
                         sessions, step_size, n_folds_list, time_periods, reg_params, output_dir, 
                         all_sessions=False, batch_size=1):
    """
    Run the testing framework across multiple sessions with robust memory management and error handling.
    
    Parameters:
    -----------
    [previous parameters remain the same...]
    batch_size : int, optional
        Number of sessions to process before forcing garbage collection
    """
    
    
    @contextmanager
    def memory_guard():
        """Context manager to ensure memory cleanup"""
        try:
            yield
        finally:
            gc.collect()
    
    def check_memory():
        """Monitor memory usage"""
        memory_use = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if memory_use > 0.9 * psutil.virtual_memory().total / 1024 / 1024:
            print(f"Warning: High memory usage detected: {memory_use:.2f} MB")
            gc.collect()
        return memory_use
    
    def save_checkpoint(session_id, results, checkpoint_dir):
        """Save intermediate results"""
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_session_{session_id}.npz')
        np.savez_compressed(checkpoint_file, results=results)
    
    try:
        # Session selection
        if all_sessions:
            selected_sessions = list(range(1, total_sessions + 1))
        else:
            selected_sessions = get_session_selection(total_sessions, step_size)
        
        # Create main results directory with timestamp
        main_results_dir = os.path.join(output_dir, f'multi_session_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(main_results_dir, exist_ok=True)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(main_results_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save analysis parameters
        analysis_params = {
            'total_sessions': total_sessions,
            'step_size': step_size,
            'selected_sessions': selected_sessions,
            'n_folds_list': n_folds_list,
            'time_periods': {k: list(v) for k, v in time_periods.items()},
            'reg_params': list(reg_params),
            'subject_id': subject_id,
            'rate_bin_params': rate_bin_params,
            'train_test_ratio': train_test_ratio,
            'reshape_method': reshape_method,
            'start_time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        save_analysis_metadata(main_results_dir, analysis_params)
        
        # Process sessions in batches
        for batch_start in range(0, len(selected_sessions), batch_size):
            with memory_guard():
                batch_sessions = selected_sessions[batch_start:batch_start + batch_size]
                print(f"\nProcessing batch: {batch_sessions}")
                
                for idx in batch_sessions:
                    session_id = sessions[idx]
                    print(f"\nProcessing session {session_id}")
                    check_memory()  # Monitor memory usage
                    
                    try:
                        # Create session-specific directory
                        session_results_dir = os.path.join(main_results_dir, f'session_{session_id}/{reshape_method}')
                        os.makedirs(session_results_dir, exist_ok=True)
                        
                        # Load and process session data
                        with memory_guard():
                            residuals_dict, _, rate_bin_centers = lps.load_and_preprocess_session(
                                session_id, hemiswap, reshape_method
                            )
                        
                        all_results = {}
                        for residuals, (X, Y) in residuals_dict.items():
                            print(f"  Processing residuals: {residuals}")
                            all_results[residuals] = {}
                            residual_X, residual_Y = residuals.split('/')
                            
                            for n_folds in n_folds_list:
                                all_results[residuals][f'folds_{n_folds}'] = {}
                            
                            for time_period, (start, end) in time_periods.items():
                                print(f"    Processing time period: {time_period}")
                                check_memory()
                                
                                with memory_guard():
                                    # Extract time window data
                                    target_bins = np.where((rate_bin_centers >= start) & (rate_bin_centers <= end))[0]
                                    X_time_win = X[:, :, target_bins]
                                    Y_time_win = Y[:, :, target_bins]
                                    
                                    # Reshape data according to method
                                    if reshape_method == 'trial_averaged':
                                        X_reshaped = np.mean(X_time_win, axis=2)
                                        Y_reshaped = np.mean(Y_time_win, axis=2)
                                    elif reshape_method == 'trial_concatenated':
                                        X_reshaped, Y_reshaped = lps.trial_concatenated_residuals(X_time_win, Y_time_win)
                                    else:
                                        raise ValueError('reshape_method must be either "trial_averaged" or "trial_concatenated"')
                                    
                                    # Clear unnecessary arrays
                                    del X_time_win, Y_time_win
                                    
                                    for n_folds in n_folds_list:
                                        title = f'{n_folds}_folds_{residual_X}-{residual_Y}-{time_period}'
                                        folds_results = cca.perform_cca(
                                            X_reshaped, Y_reshaped,
                                            residual_X, residual_Y,
                                            title, session_results_dir,
                                            cv=n_folds, reg=True,
                                            reg_params=reg_params
                                        )
                                        
                                        all_results[residuals][f'folds_{n_folds}'][time_period] = folds_results
                                        
                                    # Clear reshaped arrays
                                    del X_reshaped, Y_reshaped
                            
                            # Save checkpoint after each residual type
                            save_checkpoint(session_id, all_results, checkpoint_dir)
                        
                        # Save final results for this session
                        results_file = os.path.join(session_results_dir, f'session_{session_id}_results_{reshape_method}.npz')
                        np.savez_compressed(results_file, results=all_results)
                        
                        # Clean up
                        del all_results
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Error processing session {session_id}: {str(e)}")
                        # Save error information
                        error_file = os.path.join(session_results_dir, f'error_log_{session_id}.txt')
                        with open(error_file, 'w') as f:
                            f.write(f"Error processing session {session_id}: {str(e)}")
                        continue
                
                # Force garbage collection after each batch
                gc.collect()
        
        # Clean up checkpoints after successful completion
        for checkpoint_file in os.listdir(checkpoint_dir):
            os.remove(os.path.join(checkpoint_dir, checkpoint_file))
        os.rmdir(checkpoint_dir)
        
        # Save completion timestamp
        with open(os.path.join(main_results_dir, 'completion_time.txt'), 'w') as f:
            f.write(datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        return main_results_dir
    
    except Exception as e:
        print(f"Critical error in multi-session processing: {str(e)}")
        # Save global error information
        error_file = os.path.join(main_results_dir, 'critical_error_log.txt')
        with open(error_file, 'w') as f:
            f.write(f"Critical error: {str(e)}")
        raise