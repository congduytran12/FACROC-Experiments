import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad


def interpolate_roc_fun(perf_in, n_grid=40000):
    x_vals = perf_in.fpr
    y_vals = perf_in.tpr
    
    if len(x_vals) != len(y_vals):
        raise ValueError("FPR and TPR arrays must have the same length")
    
    # sort by x_vals to ensure proper interpolation
    idx = np.argsort(x_vals)
    x_vals = x_vals[idx]
    y_vals = y_vals[idx]
    
    # interpolation function
    interp_func = interp1d(x_vals, y_vals, kind='linear', bounds_error=False, fill_value=(0, 1))
    
    # generate new x values
    x_new = np.linspace(0, 1, n_grid)
    y_new = interp_func(x_new)
    
    return {'x': x_new, 'y': y_new}


def facroc_plot(non_protected_roc, protected_roc, non_protected_group_name=None,
               protected_group_name=None, fout=None, facroc_vals=None):
    # graph parameters
    non_protected_color = "red"
    protected_color = "blue"
    non_protected_group_label = non_protected_group_name
    protected_group_label = protected_group_name
    
    # create figure
    plt.figure(figsize=(4, 4))
    
    # title with FACROC value
    if facroc_vals is not None:
        plt.title(f"FACROC = {round(facroc_vals, 4)}")
    
    # ROC curves
    plt.plot(non_protected_roc['x'], non_protected_roc['y'], color=non_protected_color, 
             linestyle='-', linewidth=1.5)
    plt.plot(protected_roc['x'], protected_roc['y'], color=protected_color, 
             linestyle='-', linewidth=1.5)
    
    # fill area between curves
    plt.fill(
        np.append(non_protected_roc['x'], protected_roc['x'][::-1]),
        np.append(non_protected_roc['y'], protected_roc['y'][::-1]),
        color='gray', alpha=0.5
    )
    
    # add label and legend
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.legend([non_protected_group_label, protected_group_label], loc='lower right')
    
    # save plot
    if fout is not None:
        plt.savefig(fout, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_facroc(aucc_result_protected, aucc_result_non_protected, 
                  protected_attribute="Gender", protected="Female", non_protected="Male", 
                  show_plot=True, filename=None):
    # check input types
    if not hasattr(aucc_result_protected, 'aucc') or not hasattr(aucc_result_non_protected, 'aucc'):
        raise ValueError('Input objects must be AUCC objects with return_rates=True')
    
    # init FACROC value
    fr = 0
    
    # interpolate ROC curves
    non_protected_roc_fun = interpolate_roc_fun(aucc_result_non_protected)
    protected_roc_fun = interpolate_roc_fun(aucc_result_protected)
    
    # function approximation for difference between curves
    x_vals = non_protected_roc_fun['x']
    y_diff = non_protected_roc_fun['y'] - protected_roc_fun['y']
    f1 = interp1d(x_vals, y_diff, kind='linear', bounds_error=False, fill_value=0)
    
    # absolute difference
    def f2(x):
        return abs(f1(x))
    
    # compute area between curves
    slice_val, _ = quad(f2, 0, 1, limit=10000)
    fr += slice_val
    
    # plot 
    if show_plot:
        output_filename = filename
        facroc_plot(non_protected_roc_fun, protected_roc_fun, 
                   non_protected, protected, 
                   fout=output_filename, facroc_vals=fr)
    
    return fr


def FACROC(aucc_result1, aucc_result2, protected_attribute="Gender", 
          protected="Female", non_protected="Male", 
          show_plot=True, r_line=True, sample=True, sample_size=500, 
          size=1, line_type=1):
    # check input types
    if not hasattr(aucc_result1, 'aucc') or not hasattr(aucc_result2, 'aucc'):
        raise ValueError('Input objects must be AUCC objects with return_rates=True')
    
    # create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # sample points
    if sample and sample_size < len(aucc_result1.fpr):
        # include first and last points
        sample_idx1 = np.sort(np.concatenate([
            [0], 
            np.random.choice(np.arange(1, len(aucc_result1.fpr)-1), sample_size), 
            [len(aucc_result1.fpr)-1]
        ]))
        fpr1 = aucc_result1.fpr[sample_idx1]
        tpr1 = aucc_result1.tpr[sample_idx1]
    else:
        fpr1 = aucc_result1.fpr
        tpr1 = aucc_result1.tpr
    
    if sample and sample_size < len(aucc_result2.fpr):
        sample_idx2 = np.sort(np.concatenate([
            [0], 
            np.random.choice(np.arange(1, len(aucc_result2.fpr)-1), sample_size), 
            [len(aucc_result2.fpr)-1]
        ]))
        fpr2 = aucc_result2.fpr[sample_idx2]
        tpr2 = aucc_result2.tpr[sample_idx2]
    else:
        fpr2 = aucc_result2.fpr
        tpr2 = aucc_result2.tpr
    
    # plot curves
    linestyle = '-' if line_type == 1 else '--'
    ax.plot(fpr1, tpr1, color='red', linewidth=size, linestyle=linestyle, label=protected)
    ax.plot(fpr2, tpr2, color='blue', linewidth=size, linestyle=linestyle, label=non_protected)
    
    # add diagonal reference line
    if r_line:
        ax.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
    
    # configure plot
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(title=protected_attribute, loc=(0.8, 0.2))
    ax.set_facecolor('white')
    ax.grid(alpha=0.3)
    
    # display plot
    if show_plot:
        plt.show()
    
    return fig
