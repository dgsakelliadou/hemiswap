from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
import numpy as np


import numpy as np
import scipy.linalg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg
from sklearn.model_selection import KFold
from tqdm import tqdm
import os

def regularized_cca(X_train, Y_train, X_test, Y_test,residual_X,residual_Y,cv=False,reg_param=0.0): 
    """Perform regularized CCA on training data and transform test data.
    To be used with CV.
    
    """
    n_features_X = X_train.shape[1]
    n_features_Y = Y_train.shape[1]
    
    # Compute covariance matrices on the training set
    C_xx = np.cov(X_train, rowvar=False)
    C_yy = np.cov(Y_train, rowvar=False)
    C_xy = np.cov(X_train, Y_train, rowvar=False)[:n_features_X, n_features_X:]
    #print(reg_param)
    # Add regularization
    rx = reg_param * np.trace(C_xx) / len(C_xx)
    ry = reg_param * np.trace(C_yy) / len(C_yy)
    C_xx_reg = C_xx + rx * np.eye(n_features_X)
    C_yy_reg = C_yy + ry * np.eye(n_features_Y)
    
    # Compute regularized correlation matrices
    R_xx = linalg.inv(linalg.sqrtm(C_xx_reg))
    R_yy = linalg.inv(linalg.sqrtm(C_yy_reg))
    K = R_xx @ C_xy @ R_yy
    
    # SVD
    U, s, Vh = linalg.svd(K, full_matrices=False)
    V = Vh.T
    
    # Compute canonical weights
    W_x = R_xx @ U
    W_y = R_yy @ V
    
    # Transform data - on test set 
    X_c = X_test @ W_x
    Y_c = Y_test @ W_y
    
    # Compute loadings and cross-loadings
    X_loadings = np.corrcoef(X_test, X_c, rowvar=False)[:n_features_X, n_features_X:]
    Y_loadings = np.corrcoef(Y_test, Y_c, rowvar=False)[:n_features_Y, n_features_Y:]
    X_cross_loadings = np.corrcoef(X_test, Y_c, rowvar=False)[:n_features_X, n_features_X:]
    Y_cross_loadings = np.corrcoef(Y_test, X_c, rowvar=False)[:n_features_Y, n_features_Y:]

    if cv:
        results = { 
            'weights_X': W_x,
            'weights_Y': W_y,
            'X_loadings': X_loadings,
            'Y_loadings': Y_loadings,
            'correlations': [np.corrcoef(X_c[:,i], Y_c[:,i])[0,1] for i in range(X_c.shape[1])],
            'optimal_reg': reg_param
        }


    else:
        results = {
            'residual_X': residual_X,
            'residual_Y': residual_Y,
            'X_canonical': X_c,
            'Y_canonical': Y_c,
            'weights_X': W_x,
            'weights_Y': W_y,
            'correlations': [np.corrcoef(X_c[:,i], Y_c[:,i])[0,1] for i in range(X_c.shape[1])],
            'X_loadings': X_loadings,
            'Y_loadings': Y_loadings,
            'X_cross_loadings': X_cross_loadings,
            'Y_cross_loadings': Y_cross_loadings,
            'optimal_reg': reg_param
        }


    
    return results


def grid_search_cca(X_train, Y_train,X_test,Y_test,residual_X, residual_Y,title,output_dir,show=False,reg_params=None):
    """Perform grid search over regularization parameters."""

    
    if reg_params is None:
        reg_params = np.logspace(-6, 2, 100)
    
    correlations = []
    for reg in reg_params:
        results = regularized_cca(X_train, Y_train, X_test, Y_test,residual_X,residual_Y,cv=False,reg_param=reg)
        r2 = results['correlations'][0] ** 2
        correlations.append(r2)

    
    plt.figure(figsize=(10, 6))
    plt.semilogx(reg_params, correlations, 'b-', marker='o')
    plt.grid(True)
    plt.xlabel('Regularization Parameter')
    plt.ylabel('First Canonical Dimension Correlation (R^2)')
    plt.title(f'{title}-CCA Performance vs Regularization')
    
    optimal_idx = np.argmax(correlations)
    optimal_reg = reg_params[optimal_idx]
    plt.axvline(x=optimal_reg, color='r', linestyle='--', 
                label=f'Optimal λ = {optimal_reg:.2e}')
    plt.legend()
    plots_dir = os.path.join(output_dir, 'grid_regularization/plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f'{plots_dir}/{title}_grid_search.png')
    plt.close()
    
   
    '''
    if show:
        plt.show()
    '''
    
    return optimal_reg, correlations


def perform_cca(X, Y, residual_X, residual_Y, title, output_dir,cv,reg=False,reg_params=None):
    if cv == 1:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        n_components = min(X_train.shape[1], Y_train.shape[1])
        if reg:
            optimal_reg,_ = grid_search_cca(X_train, Y_train, X_test, Y_test, residual_X, residual_Y, f'{title}', output_dir,reg_params=reg_params)
            print('optimal_reg:',optimal_reg)
            print('using regularized cca')
            results = regularized_cca(X_train, Y_train, X_test, Y_test,residual_X,residual_Y,cv=False,reg_param=optimal_reg)
        else:
            cca = CCA(n_components=n_components)
            cca.fit(X_train, Y_train)
            X_c, Y_c = cca.transform(X_test, Y_test)
            results = {
                'residual_X': residual_X,
                'residual_Y': residual_Y,
                'weights_X': cca.x_weights_,
                'weights_Y': cca.y_weights_,
                'X_loadings': cca.x_loadings_,
                'Y_loadings': cca.y_loadings_,
                'X_canonical': X_c,
                'Y_canonical': Y_c,
                'X_cross_loadings': np.corrcoef(X.T, Y_c.T)[:X.shape[1], X.shape[1]:],
                'Y_cross_loadings': np.corrcoef(Y.T, X_c.T)[:Y.shape[1], Y.shape[1]:],
                'correlations': [np.corrcoef(X_c[:,i], Y_c[:,i])[0,1] for i in range(X_c.shape[1])]
            }
    
    elif cv > 1:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_results = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            n_components = min(X_train.shape[1], Y_train.shape[1])
            if reg:
                optimal_reg,_ = grid_search_cca(X_train, Y_train, X_test, Y_test, residual_X, residual_Y, f'{title}', output_dir, reg_params=reg_params)
                print('optimal_reg:',optimal_reg)
                fold_results = regularized_cca(X_train, Y_train, X_test, Y_test,residual_X,residual_Y,cv=True,reg_param=optimal_reg)
            else:
                cca = CCA(n_components=n_components)
                cca.fit(X_train, Y_train)
                X_c, Y_c = cca.transform(X_test, Y_test)
                
                fold_results = {
                    'weights_X': cca.x_weights_,
                    'weights_Y': cca.y_weights_,
                    'X_loadings': cca.x_loadings_,
                    'Y_loadings': cca.y_loadings_,
                    'correlations': [np.corrcoef(X_c[:,i], Y_c[:,i])[0,1] for i in range(X_c.shape[1])]
                }
            cv_results.append(fold_results)
        
        # Average the results across folds
        results = {
            'residual_X': residual_X,
            'residual_Y': residual_Y,
            'weights_X': np.mean([r['weights_X'] for r in cv_results], axis=0),
            'weights_Y': np.mean([r['weights_Y'] for r in cv_results], axis=0),
            'X_loadings': np.mean([r['X_loadings'] for r in cv_results], axis=0),
            'Y_loadings': np.mean([r['Y_loadings'] for r in cv_results], axis=0),
            'correlations': np.mean([r['correlations'] for r in cv_results], axis=0),
            'cv_std_correlations': np.std([r['correlations'] for r in cv_results], axis=0)
        }
        
    
    else:  # cv == 0
        n_components = min(X.shape[1], Y.shape[1])
        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(X, Y)
        results = {
            'residual_X': residual_X,
            'residual_Y': residual_Y,
            'weights_X': cca.x_weights_,
            'weights_Y': cca.y_weights_,
            'X_loadings': cca.x_loadings_,
            'Y_loadings': cca.y_loadings_,
            'X_canonical': X_c,
            'Y_canonical': Y_c,
            'X_cross_loadings': np.corrcoef(X.T, Y_c.T)[:X.shape[1], X.shape[1]:],
            'Y_cross_loadings': np.corrcoef(Y.T, X_c.T)[:Y.shape[1], Y.shape[1]:],
            'correlations': [np.corrcoef(X_c[:,i], Y_c[:,i])[0,1] for i in range(X_c.shape[1])]
        }

    return results


def bootstrap_cca_significance(X, Y, n_bootstraps=1000, alpha=0.05, reg_param=0.0):
    n_samples = X.shape[0]
    original_results = perform_cca(X, Y, reg_param=reg_param)
    original_corrs = original_results['correlations']
    bootstrap_corrs = []
    
    for _ in tqdm(range(n_bootstraps)):
        # Random sampling with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot, Y_boot = X[idx], Y[idx]
        
        # Perform CCA on bootstrap sample
        boot_results = regularized_cca(X_boot, Y_boot, reg_param=reg_param)
        bootstrap_corrs.append(boot_results['correlations'])
    
    bootstrap_corrs = np.array(bootstrap_corrs)
    confidence_intervals = np.percentile(bootstrap_corrs, 
                                      [alpha/2 * 100, (1-alpha/2) * 100], 
                                      axis=0)
    
    # Determine significant dimensions
    significant_dims = np.where(confidence_intervals[0] > 0)[0]
    min_significant_dims = len(significant_dims)
    
    return {
        'original_correlations': original_corrs,
        'bootstrap_correlations': bootstrap_corrs,
        'confidence_intervals': confidence_intervals,
        'significant_dimensions': significant_dims,
        'min_dimensions': min_significant_dims
    }