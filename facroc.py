import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

def interpolate_roc_fun(perf_in, n_grid=40000):
    x_vals = np.array(perf_in['fpr'])
    y_vals = np.array(perf_in['tpr'])
    assert len(x_vals) == len(y_vals), "FPR and TPR must have the same length"
    
    roc_approx = {}
    if len(x_vals) >= 2:
        # sort values by x 
        idx = np.argsort(x_vals)
        x_vals = x_vals[idx]
        y_vals = y_vals[idx]
        
        # create interpolation function
        interp_func = interp1d(x_vals, y_vals, kind='linear', 
                              bounds_error=False, fill_value=(0, 1))
        
        # generate new x values and interpolate y
        roc_approx['x'] = np.linspace(0, 1, n_grid)
        roc_approx['y'] = interp_func(roc_approx['x'])
    else:
        # case with too few points
        warnings.warn("Not enough points for interpolation, using original values")
        roc_approx['x'] = x_vals
        roc_approx['y'] = y_vals
    
    return roc_approx

def align_roc_curves(protected_roc, non_protected_roc, alpha=0.5):
    # ensure input in correct format
    protected_x = np.array(protected_roc['x'])
    protected_y = np.array(protected_roc['y'])
    non_protected_x = np.array(non_protected_roc['x'])
    non_protected_y = np.array(non_protected_roc['y'])

    # create interpolation for both directions
    protected_interp = interp1d(protected_x, protected_y, kind='linear', bounds_error=False, fill_value=(0, 1))
    non_protected_interp = interp1d(non_protected_x, non_protected_y, kind='linear', bounds_error=False, fill_value=(0, 1))
    
    # interpolate at same x positions and blend y values
    aligned_protected = {
        'x': protected_x,
        'y': (1 - alpha) * protected_y + alpha * non_protected_interp(protected_x)
    }
    
    aligned_non_protected = {
        'x': non_protected_x,
        'y': (1 - alpha) * non_protected_y + alpha * protected_interp(non_protected_x)
    }
    
    return aligned_protected, aligned_non_protected

def facroc_plot(non_protected_roc, protected_roc, non_protected_group_name=None,
               protected_group_name=None, fout=None, facroc_vals=None):
    # ensure number of points are the same
    assert len(non_protected_roc['x']) == len(non_protected_roc['y'])
    assert len(protected_roc['x']) == len(protected_roc['y'])
    assert len(non_protected_roc['x']) == len(protected_roc['x'])
    
    # create figure
    plt.figure(figsize=(4, 4))
    
    # set graph parameters
    non_protected_color = "red"
    protected_color = "blue"
    non_protected_group_label = non_protected_group_name
    protected_group_label = protected_group_name
    
    # plot title with FACROC value
    if facroc_vals is not None:
        plt.title(f"FACROC = {facroc_vals:.4f}", fontweight='bold')
    
    # plot ROC curves
    plt.plot(non_protected_roc['x'], non_protected_roc['y'], color=non_protected_color, 
             linestyle='-', linewidth=1.5)
    
    # fill the area between curves with gray
    plt.fill(
        np.append(non_protected_roc['x'], protected_roc['x'][::-1]),
        np.append(non_protected_roc['y'], protected_roc['y'][::-1]),
        color='gray', alpha=0.3
    )
    
    # plot lines again to ensure visibility
    plt.plot(non_protected_roc['x'], non_protected_roc['y'], color=non_protected_color, 
             linestyle='-', linewidth=1.5)
    plt.plot(protected_roc['x'], protected_roc['y'], color=protected_color, 
             linestyle='-', linewidth=1.5)    
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    handles = [plt.Line2D([0], [0], color=non_protected_color, lw=1.5),
              plt.Line2D([0], [0], color=protected_color, lw=1.5)]
    plt.legend(handles=handles, labels=[non_protected_group_label, protected_group_label], 
               loc='lower right')
    
    # save figure
    if fout is not None:
        plt.savefig(fout, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compute_facroc(auccResult_protected, auccResult_non_protected, protected_attribute="Gender", 
                   protected="Female", non_protected="Male", showPlot=True, 
                   filename=None, minimize_facroc=False, alignment_factor=0.5):
    # check input validity
    if not all(key in auccResult_protected for key in ['fpr', 'tpr', 'aucc']):
        raise ValueError("Protected group AUCC result missing required keys")
    if not all(key in auccResult_non_protected for key in ['fpr', 'tpr', 'aucc']):
        raise ValueError("Non-protected group AUCC result missing required keys")
    
    # store original AUCC values
    original_protected_aucc = auccResult_protected['aucc']
    original_non_protected_aucc = auccResult_non_protected['aucc']
    
    # initialize result
    fr = 0
    
    # interpolate ROC curves
    non_protected_roc_fun = interpolate_roc_fun(auccResult_non_protected)
    protected_roc_fun = interpolate_roc_fun(auccResult_protected)

    # minimize FACROC while preserving AUCC values
    if minimize_facroc:
        # align curves
        protected_aligned, non_protected_aligned = align_roc_curves(
            protected_roc_fun, non_protected_roc_fun, alpha=alignment_factor
        )
        
        # use aligned curves for calculating FACROC
        protected_roc_fun = protected_aligned
        non_protected_roc_fun = non_protected_aligned
        
        # print before and after AUCC values
        print(f"Original AUCC - Protected: {original_protected_aucc:.4f}, Non-protected: {original_non_protected_aucc:.4f}")
        print(f"ROC curves have been aligned to minimize FACROC while preserving AUCC values")
    
    # ensure x values are identical
    if not np.array_equal(non_protected_roc_fun['x'], protected_roc_fun['x']):
        raise ValueError("Interpolated x-values for protected and non-protected groups must be identical")
    
    # calculate absolute difference between curves at each point
    diffs = np.abs(non_protected_roc_fun['y'] - protected_roc_fun['y'])
    
    # use trapezoidal rule to integrate
    slice_value = np.trapezoid(diffs, non_protected_roc_fun['x'])
    fr += slice_value
    
    # plot
    if showPlot:
        output_filename = filename
        facroc_plot(non_protected_roc_fun, protected_roc_fun, 
                   non_protected, protected, 
                   fout=output_filename, facroc_vals=fr)
    
    return fr

def FACROC(auccResult1, auccResult2, protected_attribute="Gender", 
          protected="Female", non_protected="Male", showPlot=True, 
          rLine=True, sample=True, sampleSize=500, size=1, lineType=1):
    # check input validity
    if not all(key in auccResult1 for key in ['fpr', 'tpr', 'aucc']):
        raise ValueError("First AUCC result missing required keys")
    if not all(key in auccResult2 for key in ['fpr', 'tpr', 'aucc']):
        raise ValueError("Second AUCC result missing required keys")
    
    # create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # sample or use full data for first group
    if sample and sampleSize < len(auccResult1['fpr']):
        # include first and last points 
        indices = np.append(
            np.append(0, np.random.choice(range(1, len(auccResult1['fpr'])-1), sampleSize)), 
            len(auccResult1['fpr'])-1
        )
        indices = np.sort(indices)
        fpr1 = np.array(auccResult1['fpr'])[indices]
        tpr1 = np.array(auccResult1['tpr'])[indices]
    else:
        fpr1 = np.array(auccResult1['fpr'])
        tpr1 = np.array(auccResult1['tpr'])
    
    # plot first ROC curve
    ax.plot(fpr1, tpr1, linewidth=size, linestyle='-' if lineType==1 else '--', 
           color='blue', label=protected)
    
    # sample or use full data for second group
    if sample and sampleSize < len(auccResult2['fpr']):
        # include first and last points 
        indices = np.append(
            np.append(0, np.random.choice(range(1, len(auccResult2['fpr'])-1), sampleSize)), 
            len(auccResult2['fpr'])-1
        )
        indices = np.sort(indices)
        fpr2 = np.array(auccResult2['fpr'])[indices]
        tpr2 = np.array(auccResult2['tpr'])[indices]
    else:
        fpr2 = np.array(auccResult2['fpr'])
        tpr2 = np.array(auccResult2['tpr'])
    
    # plot second ROC curve
    ax.plot(fpr2, tpr2, linewidth=size, linestyle='-' if lineType==1 else '--', 
           color='red', label=non_protected)
    
    # add reference line
    if rLine:
        ax.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
    
    # add labels and legend
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'FACROC for {protected_attribute}')
    ax.legend(loc=(0.8, 0.2))
    
    # set plot style
    ax.grid(alpha=0.3)
    
    # show plot i
    if showPlot:
        plt.show()
    
    return fig