from spynal.matIO import loadmat
from spynal.spikes import psth
import numpy as np
import os
from typing import Dict, List, Tuple, Any

class SessionLoader:
    """Handles loading and processing of neural session data."""
    
    def __init__(self, hemiswap):
        self.hemiswap = hemiswap
        self.variables = [
            'ain', 'ainSchema', 'analogChnlInfo', 'electrodeInfo', 'eventSchema',
            'fileInfo', 'lfp', 'lfpSchema', 'sessionInfo', 'spikeChnlInfo',
            'spikeTimes', 'spikeTimesSchema', 'trialInfo', 'unitInfo'
        ]

    def load_sessions(self, subject, step: int = 8) -> Dict[str, Dict]:
        """
        Load multiple sessions with specified step size.
        
        Args:
            step: Number of sessions to skip between loads
            
        Returns:
            Dictionary of processed session data
        """
        sessions = {}
        
        for session_id in self.hemiswap.sessions['full'][::step]:
            if subject not in session_id:
                continue    
            try:
                print(f"Processing session {session_id}")
                sessions[session_id] = self._process_single_session(session_id)
            except Exception as e:
                print(f"Error processing session {session_id}: {e}")
                continue
                
        return sessions

    def _process_single_session(self, session_id: str) -> Dict[str, Any]:
        """Process a single session's data."""
        filepath = os.path.join(self.hemiswap.loadDir, f'{session_id}.mat')
        
        # Load mat file data
        data = self._load_mat_file(filepath)
        
        # Extract trial indices
        trial_indices = self._get_trial_indices(data['trialInfo'])
        
        # Get hemisphere indices
        hemi_indices = self._get_hemisphere_indices(data['unitInfo'])
        
        # Process spike data
        spike_data = self._process_spike_data(
            data['spikeTimes'], 
            trial_indices, 
            hemi_indices
        )
        
        # Combine all data
        return {
            **data,
            **trial_indices,
            **spike_data
        }

    def _load_mat_file(self, filepath: str) -> Dict[str, Any]:
        """Load MAT file with specified variables."""
        return dict(zip(
            self.variables,
            loadmat(
                filepath,
                variables=self.variables,
                typemap={'trialInfo': 'DataFrame'},
                verbose=True
            )
        ))

    @staticmethod
    def _get_trial_indices(trial_info) -> Dict[str, np.ndarray]:
        """Extract trial indices based on conditions."""
        conditions = {
            'noswap_trials_right': ('right', False),
            'noswap_trials_left': ('left', False),
            'swap_trials_right': ('right', True),
            'swap_trials_left': ('left', True)
        }
        
        return {
            key: np.where(
                (trial_info['sampleHemifield'] == hemi) & 
                (trial_info['isSwap'] == is_swap)
            )[0]
            for key, (hemi, is_swap) in conditions.items()
        }

    @staticmethod
    def _get_hemisphere_indices(unit_info) -> Dict[str, np.ndarray]:
        """Get indices for each hemisphere."""
        return {
            'right_hemi': np.where(unit_info['hemisphere'] == 'right')[0],
            'left_hemi': np.where(unit_info['hemisphere'] == 'left')[0]
        }

    @staticmethod
    def _process_spike_data(spike_times: np.ndarray, 
                           trial_indices: Dict[str, np.ndarray],
                           hemi_indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process spike data for different conditions and hemispheres."""
        spike_data = {}
        
        # Process no-swap trials for both hemispheres
        for start_hemi in ['right', 'left']:
            trials = trial_indices[f'noswap_trials_{start_hemi}']
            spikes = np.squeeze(spike_times[trials, :])
            
            # Get spikes for both hemispheres
            for record_hemi in ['right', 'left']:
                hemi_idx = hemi_indices[f'{record_hemi}_hemi']
                key = f'{start_hemi}_{record_hemi}_hemi_trials'
                spike_data[key] = np.squeeze(spikes[:, hemi_idx])
        
        return spike_data

def load_neural_data(hemiswap, subject, step: int = 7) -> Dict[str, Dict]:
    """
    Main function to load neural data.
    
    Args:
        hemiswap: HemiSwap constants object
        step: Number of sessions to skip between loads
    
    Returns:
        Dictionary of processed session data
    """
    loader = SessionLoader(hemiswap)
    return loader.load_sessions(subject,step)