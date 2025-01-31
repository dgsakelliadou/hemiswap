# -*- coding: utf-8 -*-
"""
hemiSwap_consts.py  Constants for hemiSwap dataset

Created on Fri May  3 16:17:51 2019

@author: sbrincat
"""
import os
import re
from datetime import datetime
import numpy as np
import numpy.matlib as npm
import pandas as pd
import xarray as xr
import seaborn as sns # TEMP remove '.apionly' when upgrade to Seaborn 0.8
#import seaborn.apionly as sns # TEMP remove '.apionly' when upgrade to Seaborn 0.8




# =============================================================================
# hemiSwap dataset constants
# =============================================================================
class HemiSwap_consts():
    
    def __init__(self,hostname,subject=None):
        super().__init__()
        self.hostname = hostname
        # Set up default directories for loading data, saving results
        if hostname == 'miller-lab-3': # Lab Linux desktop
            self.loadDir        = r'/mnt/common/datasets/hemiSwap/mat'
            self.loadDirMeta    = r'/mnt/common/datasets/hemiSwap/metadata'
            self.saveDir        = r'/mnt/dgsak/data/hemiSwap/results'
        elif hostname == 'miller-lab-2':   # Lab Windows desktop
            self.loadDir        = r'\\millerdata.mit.edu\common\datasets\hemiSwap\mat'
            self.loadDirMeta    = r'\\millerdata.mit.edu\common\datasets\hemiSwap\metadata'
            self.saveDir        = r'\\millerdata.mit.edu\sbrincat\data\hemiSwap\results'
        elif hostname == 'princess':       # Lab Linux server
            self.loadDir        = r'/mnt/common/datasets/hemiSwap/mat'
            self.loadDirMeta    = r'/mnt/common/datasets/hemiSwap/metadata'
            self.saveDir        = r'/mnt/dgsak/data/hemiSwap/results'

        if subject is not None:
            assert subject.lower() in ['tiergan','edith'], \
                "Unsupported subject name '%s'. Must be either 'Tiergan' or 'Edith'" % subject

        # Load Excel sheet with session metadata (specific task run, parameters, etc.)
        filename = os.path.join(self.loadDirMeta,'hemiSwapRecordingLog.xlsx')
        self.sessionTable = pd.read_excel(filename, dtype={'usable':bool}, engine='openpyxl')

        # Lists of sessions to include in analysis by default
        # 'test' -- set of "good" example sessions to pilot/test analysis on
        # 'full' -- full set of usable sessions in dataset
        self.sessions   = {'test': ['hemiSwap_Tiergan_20170726','hemiSwap_Tiergan_20170727','hemiSwap_Tiergan_20170801',
                                    'hemiSwap_Tiergan_20170920','hemiSwap_Tiergan_20170926',
                                    'hemiSwap_Edith_20180530','hemiSwap_Edith_20180613','hemiSwap_Edith_20180614',
                                    'hemiSwap_Edith_20180615','hemiSwap_Edith_20180619','hemiSwap_Edith_20180621','hemiSwap_Edith_20180622'],
                           'full': self.sessionTable['session'][self.sessionTable['usable']].tolist()}

        # Regular expression search pattern for session ID format '<task>_<Subject>_<YYYYMMDD>'
        self.sessionRegexp = '(\w+)_(\w+)_(\w+)'
        # Lambda to convert sessionID string -> task name string (always 'hemiSwap')
        self.session2task   = lambda session: re.search(self.sessionRegexp,session)[1]
        # Lambda to convert sessionID string -> subject name string ('Tiergan'|'Edith')
        self.session2subject= lambda session: re.search(self.sessionRegexp,session)[2]
        # Lambda to convert sessionID string -> datetime object
        self.session2date   = lambda session: datetime.strptime(re.search(self.sessionRegexp,session)[3],'%Y%m%d')

        # Full time range of extracted trial data (s rel to sampleOn)
        self.timeRange  = [-1.75,4.5]

        self.smpRate    = 1000 # Sampling rate for analog signals (Hz)

        self._timeIndex = []    # (nTimePts,) Time dim index of raw data (s from ref)


        # Array electrode spacing (in mm)
        self.electrodeSpacing = 0.4

        self.elec2loc   = self.set_elec2loc()
        self.elec2area  = self.set_elec2area(subject)
        self.elec2hemi  = self.set_elec2hemi(subject)


        # Times of key task trial events, for each of several reference events
        # ('sampleOn': s rel. to sample onset; 'fixation2Time': s rel. to post-saccade re-fixation)
        # Organized as dict of {String=alignEvent: [list of times]}
        self.eventTimes = {'sampleOn' : {'Fixation':[-700e-3], 'Sample':[0,700e-3],
                                         'Swap cue':[1500e-3], 'Test':[2300e-3]},
                           'fixation2Time': {'Swap cue':[-187e-3], 'Swap cue IQR':[-193e-3,-175e-3],
                                             'Post-swap fixation':[0], 'Test':[800e-3]},
                           'testOn':    {'Swap cue':[-987e-3], 'Swap cue IQR':[-993e-3,-975e-3],
                                         'Post-swap fixation':[-800e-3], 'Test':[0]}}

        # Labels for each task trial epoch, and time value to plot them at,
        # for each of several reference events
        self.epochLabels= {'sampleOn' :     {'Sample':350e-3, 'Delay 1':1100e-3},
                           'fixation2Time': {'Delay 2':400e-3, 'Test':1000e-3}}

        # Standard plotting colors for different hemiswap conditions
        # Stored as { 'colorScheme': {'cond' : [R,G,B]} } for various color schemes
        palette     = sns.color_palette("BrBG", 13) # WAS 9
        ipsiDark    = palette[0]
        ipsiBright  = palette[2]
        contraBright= palette[-3]
        contraDark  = palette[-1]
        self.plotColors = {'full' : {'noswap_contra':              contraDark, # [0.12,0.12,0.12],
                                     'noswap_left':            contraDark,
                                     'noswap_ipsi':             ipsiDark, # [0.62,0.62,0.62],
                                     'noswap_right':           ipsiDark, # [0.62,0.62,0.62],
                                     'swap_contra':                ipsiBright,
                                     'swap_left':              ipsiBright,
                                     'swap_ipsi':               contraBright,
                                     'swap_right':             contraBright,
                                     'noswap_contra-noswap_ipsi':  [0.12,0.12,0.12],
                                     'swap_contra-swap_ipsi':      [0.62,0.62,0.62],
                                     'sampleHemifield':         [0.50,0.50,0.50],
                                     'isSwap':                  [0.50,0.50,0.50],
                                     'interaction':             [0.50,0.50,0.50],
                                     'swap':                   contraBright,
                                     'noswap':                 [0.50,0.50,0.50],
                                     'swap-noswap':           contraBright,
                                     },
                           'noswap':{'noswap_contra':              contraDark,
                                     'noswap_left':            contraDark,
                                     'noswap_ipsi':             ipsiDark,
                                     'noswap_right':           ipsiDark,
                                     'noswap_contra-noswap_ipsi':  [0.12,0.12,0.12],
                                     'swap':                   contraBright,
                                     'noswap':                 [0.50,0.50,0.50],
                                     'swap-noswap':           contraBright,
                                     },
                           'swap' : {'noswap_contra':              contraDark,
                                     'noswap_left':            contraDark,
                                     'noswap_ipsi':             ipsiDark,
                                     'noswap_right':           ipsiDark,
                                     'swap_contra':                ipsiBright,
                                     'swap_left':              ipsiBright,
                                     'swap_ipsi':               contraBright,
                                     'swap_right':             contraBright,
                                     'swap_contra-swap_ipsi':      [0.62,0.62,0.62],
                                     'swap_contra-noswap_ipsi':    ipsiBright,
                                     'swap_ipsi-noswap_contra':    contraBright,
                                     'sampleHemifield':         [0.50,0.50,0.50],
                                     'isSwap':                  [0.50,0.50,0.50],
                                     'interaction':             [0.50,0.50,0.50],
                                     'swap':                   contraBright,
                                     'noswap':                 [0.50,0.50,0.50],
                                     'swap-noswap':           contraBright,
                                     },
                           'swap2wayPEV':
                                     {'sampleHemifield':        [0.00,0.00,0.62],
                                     'isSwap':                  [0.75,0.75,0.00],
                                     'interaction':             [0.00,0.50,0.00],
                                     },
                           'dirPairs':
                                    {'left2right':       contraBright,
                                     'sender2receiver':  contraBright,
                                     'right2left':       ipsiBright,
                                     'receiver2sender':  ipsiBright,
                                     'left2left':        contraDark,
                                     'sender2sender':    contraDark,
                                     'right2right':      ipsiDark,
                                     'receiver2receiver':ipsiDark,
                                     'between':         contraBright,
                                     'within':          contraDark,
                                     'within_contra':   contraDark,
                                     'within_ipsi':     ipsiDark,
                                    },
                           'dirConds':
                                    {'swap':           contraBright,
                                     'noswap':         [0.50,0.50,0.50],
                                     'swap-noswap':   contraBright,
                                     }
                           }

        # Plot markers for significance markers
        self.plotMarkers = {'noswap_contra':'.', 'swap_contra':'.', 'swap_ipsi':'.', 'noswap_ipsi':'.',
                            'noswap_contra':'.', 'swap_contra':'.', 'swap_ipsi':'.', 'noswap_ipsi':'.',
                            'noswap_left':'.', 'swap_left':'.', 'swap_right':'.', 'noswap_right':'.',
                            'noswap_contra-noswap_ipsi':'*', 'swap_contra-swap_ipsi':'*',
                            'swap_contra-noswap_ipsi':'*', 'swap_ipsi-noswap_contra':'*',
                            'sampleHemifield':'$H$','isSwap':'$S$','interaction':'$X$',
                            'sender2receiver-receiver2sender':'*', 'sender2sender-receiver2receiver':'*',
                            'swap':'.', 'noswap':'.',
                            'swap-noswap':'*','between-within':'*', 'within_contra-within_ipsi':'*'
                            }

        # Returns opposite hemifield/hemisphere from given one
        self.oppositeHemi = {'left':'right', 'right':'left',
                             'L':'R', 'R':'L',
                             'contra':'ipsi', 'ipsi':'contra',
                             'in':'out', 'out':'in'}

        # Returns complementary hemiswap condition that corresponds to same set of
        # trials when refernced to opposite hemisphere from given one
        self.complementarySwapCond = {'noswap_contra':'noswap_ipsi', 'noswap_ipsi':'noswap_contra',
                                      'swap_contra':'swap_ipsi', 'swap_ipsi':'swap_contra'}

        # Returns opposite absolute hemiswap condition (reversing left/right)
        self.oppositeSwapCond = {'noswap_left' :'noswap_right',
                                 'noswap_right':'noswap_left',
                                 'swap_left'   : 'swap_right',
                                 'swap_right'  : 'swap_left'}


    def set_elec2loc(self,*unused):
        """
        Sets mapping from electrode numbers (in set {1:256}) -> (row,col) location
        in Utah arrays. (1,1) is upper-left, (8,8) is lower-right in canonical
        orientation of array.

        Note: Same for all arrays in both Tiergan and Edith, so <subject> is
        ignored here, only retained for consistency with other methods.

        OUTPUT
        elec2loc    (256,2) ndarray of ints. Maps electrode number (-1) to location in array
                    Can be used as:  row,col = elec2loc[elec-1,:]
        """
        return npm.repmat([[8,8], [7,8], [6,8], [5,8], [4,8], [3,8], [2,8], [1,8],
                           [8,7], [7,7], [6,7], [5,7], [4,7], [3,7], [2,7], [1,7],
                           [8,6], [7,6], [6,6], [5,6], [4,6], [3,6], [2,6], [1,6],
                           [8,5], [7,5], [6,5], [5,5], [4,5], [3,5], [2,5], [1,5],
                           [8,4], [7,4], [6,4], [5,4], [4,4], [3,4], [2,4], [1,4],
                           [8,3], [7,3], [6,3], [5,3], [4,3], [3,3], [2,3], [1,3],
                           [8,2], [7,2], [6,2], [5,2], [4,2], [3,2], [2,2], [1,2],
                           [8,1], [7,1], [6,1], [5,1], [4,1], [3,1], [2,1], [1,1]], 4,1)


    def set_elec2hemi(self,subject=None):
        """
        Sets mapping from electrode numbers (in set {1:256}) -> hemisphere ('left'|'right')

        INPUT
        subject     String. Subject name ('tiergan' | 'edith' here). If given,
                    array of areas returned for given subject; otherwise dict
                    returned with arrays for both subjects
        OUTPUT
        elec2hemi   If <subject> string is given:
                        (256,) ndarray of objects (strings).
                        Maps electrode number (-1) to hemisphere
                        Can be used as:  area = elec2hemi[elec-1]
                    If no <subject> is given:
                        Dict {'subject' : (256,) ndarray (as above)}
        """
        elec2hemi = {'tiergan': np.hstack((np.tile('right',(128,)),np.tile('left',(128,)))),
                     'edith':   np.hstack((np.tile('left',(128,)),np.tile('right',(128,))))}   
                          
        if subject is not None: return elec2hemi[subject.lower()]
        else:                   return elec2hemi
                
        
    def set_elec2area(self,subject=None):
        """
        Sets mapping from electrode numbers (in set {1:256}) -> area name ('dlPFC'|'vlPFC')

        INPUT
        subject     String. Subject name ('tiergan' | 'edith' here). If given,
                    array of areas returned for given subject; otherwise dict
                    returned with arrays for both subjects
        OUTPUT
        elec2area   If <subject> string is given:
                        (256,) ndarray of objects (strings).
                        Maps electrode number (-1) to area label
                        Can be used as:  area = elec2area[elec-1]
                    If no <subject> is given:
                        Dict {'subject' : (256,) ndarray (as above)}
        """
        elec2area = {'tiergan': np.hstack((np.tile('dlPFC',(64,)), np.tile('vlPFC',(64,)),
                                           np.tile('dlPFC',(64,)), np.tile('vlPFC',(64,)))),                                           
                     'edith':   np.hstack((np.tile('vlPFC',(64,)), np.tile('dlPFC',(64,)),
                                           np.tile('vlPFC',(64,)), np.tile('dlPFC',(64,))))}
        
        if subject is not None: return elec2area[subject.lower()]
        else:                   return elec2area


    def setDefaultConfig(self,dataType,timeRange,timeMethod=None,
                         spectralMethod='wavelet',smpRate=1000,
                         alignEvents=['sampleOn','fixation2Time'], **kwargs):

        """
        Sets default configuration parameters for given neural dataType
        """
        config = super().setDefaultConfig(dataType,timeRange,timeMethod=timeMethod,
                                          spectralMethod=spectralMethod,smpRate=smpRate,
                                          **kwargs)

        # DEL
        # if 'alignEvents' in kwargs: alignEvents = kwargs['alignEvents']
        # else:                       alignEvents = ['sampleOn','fixation2Time']

        if dataType in ['muaRate','suaRate','amuaRate']:
            # For epoch-based analysis, let's standardize epoch length across delays
            timeEpochs  = {'sample' : ('sampleOn',[100e-3,600e-3]),
                           'delay1' : ('sampleOn',[1000e-3,1500e-3]),
                           'saccade': (alignEvents[-1],[-250e-3,250e-3]),
                           'delay2' : (alignEvents[-1],[300e-3,800e-3])}

            if (timeMethod == 'epoch'):
                timePts     = [0,1]     # HACK indexes of timepts for each alignEvent
                config.update({'timePts':timePts})

        elif 'lfp' in dataType:
            # Time epochs for summary plots
            timeEpochs  = {'fixation'   : ('sampleOn',[-200e-3,0]),
                           'sample'     : ('sampleOn',[100e-3,500e-3]),
                           'delay1'     : ('sampleOn',[1000e-3,1500e-3]),
                           'transfer'   : (alignEvents[-1],[-200e-3,0]),
                           'postSaccade': (alignEvents[-1],[0,300e-3]),
                           'delay2'     : (alignEvents[-1],[300e-3,800e-3])}

            # DEL?
            #timeEpochs  = {'fixation'   : ('sampleOn',[-300e-3,0]),   # WAS [-300e-3,-150e-3]
            #               'samplePeak' : ('sampleOn',[50e-3,100e-3]),
            #               'sampleDip'  : ('sampleOn',[150e-3,600e-3]),
            #               'delay1'     : ('sampleOn',[1000e-3,1500e-3]),
            #               'preSaccade' : (alignEvents[-1],[-200e-3,0]),
            #               'postSaccadeDip': (alignEvents[-1],[50e-3,300e-3]),
            #               'postSaccadeRebound': (alignEvents[-1],[400e-3,800e-3])}

            # For 'regress' method: Formula for model design matrix, in patsy/R-type notation
            # Note: C(*, Sum) is an effects-type categorical coding for given where each level
            #   is contrasted to a reference level, and they sum to 0.
            #   See: www.statsmodels.org/dev/contrasts.html
            evokedModelFormula = '1 + C(sampleObj, Sum) * C(sampleLoc, Sum) * C(fixLoc, Sum) * C(isSwap, Sum)'
            config.update({'evokedModelFormula':evokedModelFormula})

        config.update({'timeEpochs':timeEpochs})
        # DEL
        # 'timeEpochNames' : [name for name in timeEpochs.keys()],
        # 'timeEpochRefs'  : [ref for ref,tRange in timeEpochs.values()],
        # 'timeEpochRanges': [tRange for ref,tRange in timeEpochs.values()]}

        return config


    def selectTrials(self,trialInfo,criteria=None):
        """
        Default trial selection for dataset -- select correct trials, without bad timing,
        excluding trials with extremely long hemiswap saccade RTs

        selTrials = selectTrials(trialInfo)

        INPUT
        trialInfo   Dict | DataFrame. Per-trial task/behavioral metadata table

        OUTPUT
        selTrials   (nTrials,) ndarray of bools. Labels trials that meet criteria
        """
        # Implement basic criteria (correct,~badTrials) in datasetConsts
        # and combine (AND) within hemiswap saccade criterion
        if criteria is None:
            selTrials = super().selectTrials(trialInfo) & \
                        ((trialInfo['fixation2Time'] < 2) | np.isnan(trialInfo['fixation2Time']))
        else:
            selTrials = criteria(trialInfo)
        return np.asarray(selTrials)


    def swapCond2Trials(self,swapCond,hemisphere,trialInfo):
        """
        Maps swap condition string onto boolean labelling trials corresponding to it

        INPUTS
        swapCond    String. String ID for hemiSwap condition, from set:
                    'noswap_contra' : sample in contralateral hemifield, remains there
                    'noswap_ipsi'   : sample in ipsilateral hemifield, remains there
                    'swap_contra'   : sample in ipsilateral hemifield, swaped to contralateral
                    'swap_ipsi'     : sample in contralateral hemifield, swaped to ipsilateral

        hemisphere  String. Which reference hemisphere are we analyzing: 'left'|'right'

        trialInfo   DataFrame | Dict. Per-trial task metadata.  Should have columns/keys:
                    'sampleHemifield' : Hemifield of sample object ('left'|'right')
                    'isSwap' : Boolean vector labelling hemiswap trials

        RETURNS
        trialBool   (nTrials,) bool. 1's flag all trials matching given condition
        """
        # Trials where sample is in contralateral hemifield to given hemisphere
        contraTrials= trialInfo['sampleHemifield'] == self.oppositeHemi[hemisphere]
        # DEL Trials where fixation begins in ispilateral hemifield to current hemisphere
        # DEL and thus sample appears in *contralateral* hemifield
        # DEL contraTrials= (trialInfo['fixLoc'] == hemisphere[0].upper())

        # Boolean vector labelling hemiswap trials
        isSwap      = trialInfo['isSwap']

        # If trialInfo is DataFrame, convert Series -> Numpy vector
        if not isinstance(trialInfo,dict):
            contraTrials    = contraTrials.values
            isSwap          = isSwap.values

        # Trials = fixation starts and remains in reference hemifield
        #  (no saccade/swap), keeping sample and memory in same location
        if swapCond == 'noswap_contra': return contraTrials & ~isSwap
        # Trials = trials where fixation start in reference hemifield,
        #  saccades to opposite, swaping memory out of reference hemi
        elif swapCond == 'swap_ipsi':   return contraTrials & isSwap
        # Trials = trials where fixation start in opposite hemifield,
        #  saccades to reference hemi, swaping memory into reference hemi
        elif swapCond == 'swap_contra': return ~contraTrials & isSwap
        # Trials = trials where fixation start in opposite hemifield,
        #  and remains in opposite hemifield (no saccade/swap)
        elif swapCond == 'noswap_ipsi': return ~contraTrials & ~isSwap
        else:
            raise ValueError("Unsupported value set for <swapCond>: %s" % swapCond)


    def swapCondGlobal2Trials(self,swapCond,trialInfo):
        """
        Maps global/bilateral swap condition string onto boolean labelling
        trials corresponding to it

        INPUTS
        swapCond    String. String ID for hemiSwap condition, from set:
                    'noswap_left' 	: sample in left hemifield, remains there
                    'noswap_right'  : sample in right hemifield, remains there
                    'swap_left'   	: sample in right hemifield, swaped to left
                    'swap_right'  	: sample in left hemifield, swaped to right

        trialInfo   DataFrame | Dict. Per-trial task metadata.  Should have columns/keys:
                    'sampleHemifield' : Hemifield of sample object ('left'|'right')
                    'isSwap' : Boolean vector labelling hemiswap trials

        RETURNS
        trialBool   (nTrials,) bool. 1's flag all trials matching given condition
        """
        swapCond,testHemi = swapCond.split('_')
        if swapCond == 'swap':        sampleHemi = self.oppositeHemi[testHemi]
        elif swapCond == 'noswap':    sampleHemi = testHemi
        else:
            raise ValueError("Unsupported value '%s' set for <swapCond>" % swapCond)

        # Trials where sample is in given hemifield
        sampleHemiTrials = trialInfo['sampleHemifield'] == sampleHemi

        # Boolean vector labelling hemiswap trials
        isSwap = trialInfo['isSwap']
        
        # If trialInfo is DataFrame, convert Series -> Numpy vector
        if not isinstance(trialInfo,dict):
            sampleHemiTrials    = sampleHemiTrials.values
            isSwap              = isSwap.values
                    
        # Combine with swap/swap condition to get final trial selection vector
        if swapCond == 'swap':  return sampleHemiTrials & isSwap
        else:                   return sampleHemiTrials & ~isSwap


    def poolHemiCategories(self,poolCond,poolHemi='both'):
        hemis = ['left','right']
        
        if poolCond == 'noswap_contra':
            conds = ['noswap_right','noswap_left']
        elif poolCond == 'noswap_ipsi':
            conds = ['noswap_left','noswap_right']
        elif poolCond == 'swap_contra':
            conds = ['swap_right','swap_left']
        elif poolCond == 'swap_ipsi':
            conds = ['swap_left','swap_right']
            
        if poolHemi not in ['both','all']:
            hemis = np.asarray(hemis)
            conds = np.asarray(conds)
            idx = hemis == poolHemi
            hemis = hemis[idx].tolist()
            conds = conds[idx].tolist()
            
        return hemis,conds
    

    def poolChannelPairCategories(self,poolPair,poolCond):
        """
        Returns ordered lists of hemifield conds and hemisphere pair categories
        included in given pooled/hemisphere-referenced condition,pair for
        causality/directionality analyses
        """
        # Pool swap conditions together
        if poolCond == 'swap':
            # Note: conds refers to the *visual hemifield*, but
            # pairs refers to the *cortical hemisphere*, and they are opposite
            conds = ['swap_right','swap_left']

            # Directional interactions from putative 'sender' -> 'receiver'
            if poolPair in ['sender2receiver']:
                pairs = ['right2left','left2right']
            # Directional interactions from putative 'receiver' -> 'sender'
            elif poolPair in ['receiver2sender']:
                pairs = ['left2right','right2left']
            # Directional interactions btwn channels in 'sender' hemisphere
            elif poolPair in ['sender2sender']:
                pairs = ['right2right','left2left']
            # Directional interactions btwn channels in 'receiver' hemisphere
            elif poolPair in ['receiver2receiver']:
                pairs = ['left2left','right2right']

            # Non-Directional interactions between hemispheres
            elif poolPair in ['between']:
                pairs = ['between','between']
            # Non-Directional interactions btwn channels in contrateral/sender hemisphere
            elif poolPair in ['within_contra']:
                pairs = ['right','left']
            # Non-Directional interactions btwn channels in ipsilateral/receiver hemisphere
            elif poolPair in ['within_ipsi']:
                pairs = ['left','right']

            else:
                raise ValueError("Unsupported value for <poolPair>: '%s'" % poolPair)

        # Pool no-swap conditions together
        # HACK Here 'sender' = hemisphere where memory trace is stored (contralateral);
        #      'receiver' = opposite hemisphere (ipsilateral)
        elif poolCond == 'noswap':
            conds = ['noswap_right','noswap_left']

            # Directional interactions from contrateral -> ipsilateral hemisphere
            if poolPair in ['sender2receiver','contra2ipsi','in2out']:
                pairs = ['left2right','right2left']
            # Directional interactions from ipsilateral -> contrateral hemisphere
            elif poolPair in ['receiver2sender','ipsi2contra','out2in']:
                pairs = ['right2left','left2right']
            # Directional interactions btwn channels in contrateral hemisphere
            elif poolPair in ['sender2sender','contra2contra','in2in']:
                pairs = ['left2left','right2right']
            # Directional interactions btwn channels in ipsilateral hemisphere
            elif poolPair in ['receiver2receiver','ipsi2ipsi','out2out']:
                pairs = ['right2right','left2left']

            # Non-Directional interactions between hemispheres
            elif poolPair in ['between']:
                pairs = ['between','between']
            # Non-Directional interactions btwn channels in contrateral hemisphere
            elif poolPair in ['within_contra']:
                pairs = ['left','right']
            # Non-Directional interactions btwn channels in ipsilateral hemisphere
            elif poolPair in ['within_ipsi']:
                pairs = ['right','left']

            else:
                raise ValueError("Unsupported value for <poolPair>: '%s'" % poolPair)

        return pairs,conds


    def generateTargetLabels(self,target,trialInfo):
        """
        Generates vector of target labels (eg for classification) for given
        target type

        INPUTS
        target  String. Which type of target to use:
                'object'    : Sample object identity (A vs B)
                'location'  : Sample location (up vs down)
                'sample'    : All 4 conditions of object identity x location

        trialInfo   DataFrame | Dict. Per-trial task metadata.  Should have columns/keys:
                    'sampleObj' : Sample object identity
                    'sampleLoc' : Sample object location

        RETURNS
        target  (nTrials,). Integer target labels for given target type
        """
        # Predicting sample object identity (1 | 2)
        if target in ['object','sampleObj']:
            labels = trialInfo['sampleObj']
        # Predicting sample location (1 | 2)
        elif target in ['location','sampleLoc']:
            labels = trialInfo['sampleLoc']
        # Predicting all 4 sample conditions: 2 sample locations x 2 objects
        elif target in ['sample','sampleObjXsampleLoc']:
            labels = ((trialInfo['sampleLoc']-1)*2 + trialInfo['sampleObj'])
        else:
            raise ValueError("Unsupported value set for <target>: %s" % target)

        return labels.values.astype(int)

    def get_eventTimes(self,alignEvent='sampleOn'):
        """
        Returns times of task trial events, selecting values corresponding
        to given time-alignment reference event (default: 'sampleOn')
        """
        return self.eventTimes[alignEvent]

    def get_epochLabels(self,alignEvent='sampleOn'):
        """
        Returns times of task trial epochs, selecting values corresponding
        to given time-alignment reference event (default: 'sampleOn')
        """
        return self.epochLabels[alignEvent]


def stitchAlignEvents(data,axis=None,t=None,
                      alignAxis=None,alignEvents=['sampleOn','fixation2Time'],
                      timeRange=[[None,None],[0,800e-3]],tShift=[0,1500e-3]):
    """ Stitch together data across align events """
    # Extract time axis from xarray dims
    if axis is None:
        if isinstance(data,xr.DataArray) and 'time' in data.dims:
            axis = np.nonzero(np.in1d(data.dims,'time'))[0][0]
        else:
            raise ValueError("Must input value for <axis> if <data> is not type xarray with 'time' dim")

    # Extract alignAxis axis from xarray dims
    if alignAxis is None:
        if isinstance(data,xr.DataArray) and 'alignEvent' in data.dims:
            alignAxis = np.nonzero(np.in1d(data.dims,'alignEvent'))[0][0]
        else:
            raise ValueError("Must input value for <alignAxis> if <data> is not type xarray with 'alignEvent' dim")

    # Extract time index from xarray coords
    if t is None:
        if isinstance(data,xr.DataArray) and 'time' in data.dims:
            t = data.coords['time'].values
        else:
            raise ValueError("Must input value for <t> if <data> is not type xarray with 'time' dim")

    # Set defaults for time range = full time range
    for iAlign in range(2):
        if timeRange[iAlign][0] is None: timeRange[iAlign][0] = t[0]
        if timeRange[iAlign][1] is None: timeRange[iAlign][1] = t[-1]

    # Booleans for extracting time points to retain for 1st,2nd align events
    tBool = [(t >= timeRange[0][0]) & (t <= timeRange[0][1]),
             (t >= timeRange[1][0]) & (t <= timeRange[1][1])]

    # Generate new time sampling vector for full stitched-together data
    tStitch = np.hstack((t[tBool[0]]+tShift[0], t[tBool[1]]+tShift[1]))

    # Extract data aligned to each alignEvent, stitched together appropriately
    # If data is an xarray DataArray, package back into a DataArray again
    if isinstance(data,xr.DataArray):
        coords = {dim : data.coords[dim].values for dim in data.coords}
        coords['time'] = tStitch
        coords['alignEvent'] = [alignEvents[0]]
        dims = data.dims

        data = np.concatenate((data.sel({'alignEvent':[alignEvents[0]],'time':t[tBool[0]]}).values,
                               data.sel({'alignEvent':[alignEvents[1]],'time':t[tBool[1]]}).values),
                              axis=axis)
        data = xr.DataArray(data,dims=dims,coords=coords)

    else:
        data = np.concatenate((data.compress(tBool[0],axis=axis).take(0,axis=alignAxis),
                               data.compress(tBool[1],axis=axis).take(-1,axis=alignAxis)),
                               axis=axis)

    return data, tStitch

def reorgEpochsByAlignEvent(timeEpochs,alignEvents):
    """
    Reorganizes dict of timeEpochs from {epochName : (alignEvent,epochTimeRange)}
    to {alignEvent : {epochName:epochTimeRange}}
    """
    timeEpochs_byAlign = {}

    for alignEvent in alignEvents:
        timeEpochs_byAlign[alignEvent] = {}
        for epoch,(alignEvent2,tRange) in timeEpochs.items():
            if alignEvent == alignEvent2: timeEpochs_byAlign[alignEvent].update({epoch:tRange})

    return timeEpochs_byAlign

def rereferenceAnalog(data,electrodeInfo,scheme='array'):
    """
    Implements re-referencing on analog data
    """
    # TODO  Code up Laplacian, bipolar?

    if not isinstance(electrodeInfo,pd.DataFrame):
        electrodeInfo = pd.DataFrame(electrodeInfo)

    # Array-mean subtraction
    if scheme == 'array':
        # Create column labeling which array each electrode was in ~ '<hemisphere>_<area>'
        if not 'array' in electrodeInfo:
            electrodeInfo['array'] = electrodeInfo['hemisphere'] + '_' + electrodeInfo['area']
        # Find set of arrays in data
        arrays = electrodeInfo['array'].unique()

        # Remove mean across all array electrodes from each electrode in array,
        # (separately for each timepoint and trial)
        for array in arrays:
            arrayElecs = (electrodeInfo['array'] == array).values
            data[:,arrayElecs,:] -= data[:,arrayElecs,:].mean(axis=1,keepdims=True)

    else:
        raise ValueError("Value '%s' is unsupported for <scheme>. Supported values \
                          are 'array'" % scheme)

    return data