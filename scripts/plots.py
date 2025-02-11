import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import os
import load_session as ls

####### PLOTS FOR TRIAL AVERAGED RESIDUALS #######

#THIS WORKS
def plot_components_by_r2(residual_results,cv,results_dir,time_periods,show=False):
  for residual_pair, folds in residual_results.items():
        residual_X = residual_pair.split('/')[0]
        residual_Y = residual_pair.split('/')[1]
        
        #print(time_periods)
        title = f'{cv}_{residual_X}-{residual_Y}'
        fig, axes = plt.subplots(1, len(time_periods),
        figsize = (5 * len(time_periods),4),
            sharey=True)
        fig.suptitle(f"Residuals: {residual_pair}", fontsize=14, y=1.05)
        data_times = folds[cv]
        for ax, (time_period, data) in zip(axes,data_times.items()):
            #print(time_period)
            print(data)

            correlations = data['correlations']
            corr_squared = np.array(correlations)**2
            ax.bar(range(len(corr_squared)),corr_squared, color='skyblue')
            ax.set_title(f'{time_period}')
            ax.set_xlabel("Canonical Variate (sorted by strength)")
            ax.set_ylabel("$r^2$ (Squared correlation)")
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plots_dir = os.path.join(results_dir, 'plots/components_by_r2')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(f'{plots_dir}/{title}_components_by_r2.png')
        plt.close()

# i think that works as well, remains to be checked with multiple pairs
def plot_avg_r2(residual_results, title, results_dir, cv, time_periods, show=False):
    '''
    Create bar plots showing average R² correlations for each residual pair across time periods.
    This is to be used with cv = n_folds>1
    '''
    #cv = 'folds_2'
    all_means = []
    all_stds = []
    residual_pairs = list(residual_results.keys())

    for residual_pair in residual_pairs:
        means = [(residual_results[residual_pair][cv][time_period]['correlations'][0])**2 
                for time_period in time_periods]
        stds = [2 * abs(residual_results[residual_pair][cv][time_period]['correlations'][0]) *
                residual_results[residual_pair][cv][time_period]['cv_std_correlations'][0]
                for time_period in time_periods]
        
        all_means.extend(means)
        all_stds.extend(stds)

    # Set y-axis limits with margin
    y_min = np.min(np.array(all_means) - np.array(all_stds))
    y_max = np.max(np.array(all_means) + np.array(all_stds))
    y_margin = (y_max - y_min) * 0.1
    y_limits = (y_min - y_margin, y_max + y_margin)

    # Plot pairs of residuals
    for i in range(0, len(residual_pairs), 2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Average R² Across Folds', fontsize=16)
        x = np.arange(len(time_periods))
        #title = f'{cv}_{residual_pairs[i]}-{residual_pairs[i+1]}'
        
        
        # First residual pair plot
        mean_r2_1 = [(residual_results[residual_pairs[i]][cv][time_period]['correlations'][0])**2 
                    for time_period in time_periods]
        std_r2_1 = [2 * abs(residual_results[residual_pairs[i]][cv][time_period]['correlations'][0]) *
                residual_results[residual_pairs[i]][cv][time_period]['cv_std_correlations'][0]
                for time_period in time_periods]
        
        ax1.bar(x, mean_r2_1, yerr=std_r2_1, capsize=5, color='orange')
        ax1.set_title(f'Residual pair {residual_pairs[i]}')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Average R²')
        ax1.set_xticks(x)
        ax1.set_xticklabels(time_periods, rotation=45)
        ax1.set_ylim(y_limits)
        
        # Second residual pair plot (if exists)
        if i + 1 < len(residual_pairs):
            mean_r2_2 = [(residual_results[residual_pairs[i+1]][cv][time_period]['correlations'][0])**2 
                        for time_period in time_periods]
            std_r2_2 = [2 * abs(residual_results[residual_pairs[i+1]][cv][time_period]['correlations'][0]) *
                residual_results[residual_pairs[i+1]][cv][time_period]['cv_std_correlations'][0]
                for time_period in time_periods]
            
            ax2.bar(x, mean_r2_2, yerr=std_r2_2, capsize=5, color='orange')
            ax2.set_title(f'Residual pair {residual_pairs[i+1]}')
            ax2.set_xlabel('Time Period')
            ax2.set_ylabel('Average R²')
            ax2.set_xticks(x)
            ax2.set_xticklabels(time_periods, rotation=45)
            ax2.set_ylim(y_limits)
        else:
            fig.delaxes(ax2)
        plt.tight_layout()
        plots_dir = os.path.join(results_dir, 'plots/average_r2_across_folds')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(f'{plots_dir}/{title}_average_r2.png')
        plt.close()

#THIS WORKS
def plot_residual_correlations(residual_results, cv, results_dir, time_periods):
    """
    Create bar plots showing R² correlations for each residual pair across time periods.
    
    Parameters:
    residual_results (dict): Dictionary containing residual pairs and their correlation data
    time_periods (list): List of time periods to plot
    """
    # Set style for better visualization
    plt.style.use('seaborn')
    
    # Create a figure for each residual pair
    for residual_pair in residual_results.keys():
        residual_X = residual_pair.split('/')[0]
        residual_Y = residual_pair.split('/')[1]
        title = f'{cv}_{residual_X}-{residual_Y}'
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        
        # Extract R² values for each time period
        r2_values = [(residual_results[residual_pair][cv][time_period]['correlations'][0])**2
                     for time_period in time_periods]
        
        # Create bar plot
        bars = ax.bar(range(len(time_periods)), r2_values, color='royalblue', alpha=0.7)
        
        # Customize the plot
        ax.set_title(f'{cv}_1st Dimension R² Correlation: {residual_X} vs {residual_Y}', pad=20)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('R² Value')
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(time_periods)))
        ax.set_xticklabels(time_periods, rotation=45)
        
        # Set y-axis limits (since R² is between -1 and 1)
        ax.set_ylim(0, 1)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        # Add grid for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plots_dir = os.path.join(results_dir, 'plots/first_dimension_r2')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(f'{plots_dir}/{title}_first_dimension_r2.png')
        plt.close()
        #plt.show()


#THIS WORKS
def plot_condition_results(paired_results,time_periods,results_dir,cv):
    '''
    Plot comparison of R2 correlation for the 1st canonical dimension between swap and no-swap conditions
    '''
    fig, axes = plt.subplots(len(paired_results), 1, figsize=(10, 5*len(paired_results)))
    if len(paired_results) == 1:
        axes = [axes]
    
    x = np.arange(len(time_periods))

    for idx, (pair, conditions) in enumerate(paired_results.items()):
        # Calculate R² and propagate errors
        swap_corrs = [conditions['swap'][period]['correlations'][0] for period in time_periods]
        no_swap_corrs = [conditions['no-swap'][period]['correlations'][0] for period in time_periods]
        swap_corrs_std = [conditions['swap'][period]['cv_std_correlations'][0] for period in time_periods]
        no_swap_corrs_std = [conditions['no-swap'][period]['cv_std_correlations'][0] for period in time_periods]
        
        # Convert to R² and propagate errors (using error propagation formula for R² = R * R)
        swap_r2 = [r**2 for r in swap_corrs]
        no_swap_r2 = [r**2 for r in no_swap_corrs]
        swap_r2_std = [2 * abs(r) * std for r, std in zip(swap_corrs, swap_corrs_std)]
        no_swap_r2_std = [2 * abs(r) * std for r, std in zip(no_swap_corrs, no_swap_corrs_std)]
        print(swap_corrs_std[0], swap_r2_std[0])
        
        
        # Plot correlations
        axes[idx].bar(x - 0.2, swap_r2, 0.4, yerr=swap_r2_std, label='Swap', color='blue', alpha=0.6)
        axes[idx].bar(x + 0.2, no_swap_r2, 0.4, yerr=no_swap_r2_std, label='No Swap', color='red', alpha=0.6)
        
        axes[idx].set_title(f'Residual Pair: {pair}')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(time_periods, rotation=45)
        axes[idx].set_ylabel('R²')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    title = f'{cv}swap_vs_no_swap_avg_r2_1st_dim'
    plots_dir = os.path.join(results_dir, 'plots/swap_vs_no_swap_r2')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f'{plots_dir}/{title}_average_r2.png')
    plt.close()
    


###### ------- THE PLOTS BELOW SHOULD BE USED WITH CV = 1_folds ------- ######
### which is train-test split, not cross-validation
### TODO: neeed to fix swap no swap plots and add weight analysis plots (look at weight_plots.py)

def swap_no_swap_line_plot(residual_results, folder_name,time_periods,show=False):
    """
    To use with perform_cca with cv=1
    
    Create line plots showing difference in R² correlations between swap and no-swap conditions,
    for each residual pair across time periods.
    """

    df_swap_means = pd.DataFrame()
    df_no_swap_means = pd.DataFrame()
    for (residual_X, residual_Y), data in residual_results.items():
        residual_X_name = residual_X.split('-')[0] + "_" + residual_X.split('-')[1]
        residual_Y_name = residual_Y.split('-')[0] + "_" + residual_Y.split('-')[1]
        means = [data[time_period]['correlations'] for time_period in time_periods]
        df_means = pd.DataFrame(means)
        df_means['residual_pair'] = f"{residual_X_name}/{residual_Y_name}"
        df_means['type'] = 'mean'
        stds = [data[time_period]['cv_std_correlations'] for time_period in time_periods]
        df_stds = pd.DataFrame(stds)
        df_stds['residual_pair'] = f"{residual_X_name}/{residual_Y_name}"
        df_stds['type'] = 'std'
        if 'no-swap' not in residual_X:
            condition = residual_X.split('-')[2]
            df_means['condition'] = condition
            df_stds['condition'] = condition
            df_swap_means = pd.concat([df_swap_means,df_means,df_stds])
        if 'no-swap' in residual_X:
            condition = residual_X.split('-')[2] + "_" + residual_X.split('-')[3]
            df_means['condition'] = condition
            df_stds['condition'] =condition
            df_no_swap_means = pd.concat([df_no_swap_means,df_means,df_stds])
    df = pd.concat([df_swap_means,df_no_swap_means])
    diffs = []
    for i in set(df['residual_pair']):
        df_no_swap_pair = df_no_swap_means.loc[(df_no_swap_means['residual_pair']==i) & (df_no_swap_means['type']=='mean')]
        df_swap_pair = df_swap_means.loc[(df_swap_means['residual_pair']==i) & (df_swap_means['type']=='mean')]
        #print(df_no_swap_pair)
        for t in time_periods:
            diff = (df_swap_pair[t].values[0] - df_no_swap_pair[t].values[0])
            diffs.append({'residual_pair': i, 'time_period':t,'difference':diff})

    df_diff = pd.DataFrame(diffs)
    sns.relplot(
    data=df_diff,
    x='time_period',
    y='difference',
    hue='residual_pair',
    kind='line',
    marker='o'
    )
    plt.axhline(0, color='black', linestyle='--')  # If you do signed difference
    plt.xlabel("Time Period")
    plt.ylabel("Difference in R² (Top CV)")
    plt.title("Swap vs. No-Swap Differences Over Time Periods (Per Residual Pair)")
    plt.savefig(f'/home/dgsak/Documents/code/hemiswap_old/results/{folder_name}/plots/both_swap_no_swap_line_plot.png')
    plt.close()
    if show:
        plt.show()

######### - PLOTS FOR THE MULTISESSION FRAMEWORK - #########



def plot_residual_pairs_traces(mean_corrs_folds_sessions, std_corrs_folds_sessions, time_periods, residual_pairs, title, results_dir, subject_id, cv):
    '''
    Plots mean R^2 with error bands and individual session traces across time periods for each residual pair/condition.
    '''
    # Set up the plot grid
    n_pairs = len(residual_pairs)
    n_cols = 2
    n_rows = (n_pairs + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]

    x = np.arange(len(time_periods))

    # Load individual session data
    _, all_sessions_results = ls.load_multi_session_results(results_dir, subject_id)
    n_sessions = len(all_sessions_results)
    #print(n_sessions)
    colors = plt.cm.viridis(np.linspace(0, 1, n_sessions))
    for pair_idx, residual_pair in enumerate(residual_pairs):
        ax = axes[pair_idx]
        
        # Plot individual session traces
        for session_idx,(session_id, session_results) in enumerate(all_sessions_results.items()):
            session_r2 = [session_results[residual_pair][cv][tp]['correlations'][0]**2 
                         for tp in time_periods]
            ax.plot(x, session_r2, '-o', color=colors[session_idx], 
                    label=f'Session {session_id}', alpha=0.7,
                    markersize=6)
        
        # Plot mean and std
        mean_r2 = [mean_corrs_folds_sessions[residual_pair][tp] for tp in time_periods]
        std_r2 = [std_corrs_folds_sessions[residual_pair][tp] for tp in time_periods]
        
        ax.plot(x, mean_r2, '-ko', linewidth=2, label='Mean across sessions', markersize=6)
        ax.fill_between(x, 
                       np.array(mean_r2) - np.array(std_r2), 
                       np.array(mean_r2) + np.array(std_r2), 
                       color='gray', alpha=0.2)
        
        # Customize plot
        ax.set_xlabel('Time Period')
        ax.set_ylabel('1st Canonical Variate R-squared')
        ax.set_title(f'{cv}-Residual pair {residual_pair}')
        ax.set_xticks(x)
        ax.set_xticklabels(time_periods, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove empty subplots if odd number of pairs
    if n_pairs % 2 == 1 and n_pairs > 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plots_dir = os.path.join(results_dir, 'plots/sessions_R2_traces')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f'{plots_dir}/{title}_average_r2_session_traces.png')
    plt.close()
    

