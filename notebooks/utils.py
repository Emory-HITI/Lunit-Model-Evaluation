import re
from tqdm import tqdm
from PIL import Image
from itertools import combinations

import pandas as pd
import numpy as np

import scipy.stats as st
from scipy.stats import norm
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu, norm

from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import PathPatch


# Function to calculate confidence intervals for metrics using bootstrapping
def compute_metric_ci(y_true, y_score, threshold=0.1, n_bootstraps=1000, ci=0.95):
    bootstrapped_metrics = {
        'precision': [],
        'recall': [],  # This is TPR
        'f1_score': [],
        'micro_f1_score': [],
        'weighted_f1_score': [],
        'tnr': [],     # True Negative Rate
        'fnr': [],     # False Negative Rate
        'fpr': [],     # False Positive Rate
        'auc': []
    }
    
    # Original predictions
    y_pred = (y_score >= threshold).astype(int)
    
    # Bootstrapping
    for _ in range(n_bootstraps):
        # Resample indices
        indices = resample(np.arange(len(y_true)), replace=True)
        
        # Resampled data
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        y_pred_boot = y_pred[indices]
        
        # Ensure both classes are present
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        # Compute metrics
        precision_boot, recall_boot, f1_boot, _ = precision_recall_fscore_support(
            y_true_boot, y_pred_boot, average='binary', zero_division=0
        )
        
        # Calculate micro and weighted F1 scores
        _, _, micro_f1_boot, _ = precision_recall_fscore_support(
            y_true_boot, y_pred_boot, average='micro', zero_division=0
        )
        _, _, weighted_f1_boot, _ = precision_recall_fscore_support(
            y_true_boot, y_pred_boot, average='weighted', zero_division=0
        )
        
        tn_boot, fp_boot, fn_boot, tp_boot = confusion_matrix(y_true_boot, y_pred_boot).ravel()
        
        tnr_boot = tn_boot / (tn_boot + fp_boot) if (tn_boot + fp_boot) > 0 else np.nan
        fnr_boot = fn_boot / (fn_boot + tp_boot) if (fn_boot + tp_boot) > 0 else np.nan
        fpr_boot = fp_boot / (fp_boot + tn_boot) if (fp_boot + tn_boot) > 0 else np.nan
        
        # Compute AUC
        auc_boot = roc_auc_score(y_true_boot, y_score_boot)
        
        # Append to lists
        bootstrapped_metrics['precision'].append(precision_boot)
        bootstrapped_metrics['recall'].append(recall_boot)  # TPR
        bootstrapped_metrics['f1_score'].append(f1_boot)
        bootstrapped_metrics['micro_f1_score'].append(micro_f1_boot)
        bootstrapped_metrics['weighted_f1_score'].append(weighted_f1_boot)
        bootstrapped_metrics['tnr'].append(tnr_boot)
        bootstrapped_metrics['fnr'].append(fnr_boot)
        bootstrapped_metrics['fpr'].append(fpr_boot)
        bootstrapped_metrics['auc'].append(auc_boot)
    
    # Calculate mean and confidence intervals
    metrics_mean_ci = {}
    for metric in bootstrapped_metrics:
        scores = np.array(bootstrapped_metrics[metric])
        mean_score = np.mean(scores)
        lower_bound = np.percentile(scores, (1 - ci) / 2 * 100)
        upper_bound = np.percentile(scores, (1 + ci) / 2 * 100)
        metrics_mean_ci[metric] = (mean_score, lower_bound, upper_bound)
    
    return metrics_mean_ci


# Function to calculate metrics for a specific group
def compute_metrics(group_df):
    y_true = group_df['exam_pathology_binary'].values
    y_score = group_df['score'].values
    
    # Ensure both classes (0 and 1) are present
    if len(np.unique(y_true)) < 2:
        return {
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'micro_f1_score': np.nan,
            'weighted_f1_score': np.nan,
            'tn': np.nan,
            'fp': np.nan,
            'fn': np.nan,
            'tp': np.nan,
            'tnr': np.nan,
            'fnr': np.nan,
            'fpr': np.nan,
            'auc': np.nan,
            'study_count': len(group_df),
            'patient_count': len(set(group_df['empi_anon'])),            
            'total_label_0': np.sum(y_true == 0),
            'total_label_1': np.sum(y_true == 1),
        }
    
    # Threshold for predictions
    threshold = 0.1
    y_pred = (y_score >= threshold).astype(int)
    
    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics and their confidence intervals
    metrics_ci = compute_metric_ci(y_true, y_score, threshold=threshold)
    
    # Extract mean and confidence intervals
    precision_mean, precision_lower, precision_upper = metrics_ci['precision']
    recall_mean, recall_lower, recall_upper = metrics_ci['recall']
    f1_mean, f1_lower, f1_upper = metrics_ci['f1_score']
    micro_f1_mean, micro_f1_lower, micro_f1_upper = metrics_ci['micro_f1_score']
    weighted_f1_mean, weighted_f1_lower, weighted_f1_upper = metrics_ci['weighted_f1_score']
    tnr_mean, tnr_lower, tnr_upper = metrics_ci['tnr']
    fnr_mean, fnr_lower, fnr_upper = metrics_ci['fnr']
    fpr_mean, fpr_lower, fpr_upper = metrics_ci['fpr']
    auc_mean, auc_lower, auc_upper = metrics_ci['auc']
    
    metrics = {
        'precision': f"{precision_mean:.2f} ({precision_lower:.2f}, {precision_upper:.2f})",
        'recall': f"{recall_mean:.2f} ({recall_lower:.2f}, {recall_upper:.2f})",
        'f1_score': f"{f1_mean:.2f} ({f1_lower:.2f}, {f1_upper:.2f})",
        'micro_f1_score': f"{micro_f1_mean:.2f} ({micro_f1_lower:.2f}, {micro_f1_upper:.2f})",
        'weighted_f1_score': f"{weighted_f1_mean:.2f} ({weighted_f1_lower:.2f}, {weighted_f1_upper:.2f})",
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'tnr': f"{tnr_mean:.2f} ({tnr_lower:.2f}, {tnr_upper:.2f})",
        'fnr': f"{fnr_mean:.2f} ({fnr_lower:.2f}, {fnr_upper:.2f})",
        'fpr': f"{fpr_mean:.2f} ({fpr_lower:.2f}, {fpr_upper:.2f})",
        'auc': f"{auc_mean:.2f} ({auc_lower:.2f}, {auc_upper:.2f})",
        'study_count': len(group_df),
        'patient_count': len(set(group_df['empi_anon'])),
        'total_label_0': np.sum(y_true == 0),
        'total_label_1': np.sum(y_true == 1),
    }
    
    return metrics


# Function to compute AUC for each group
def compute_group_auc(df, col, score_col='score', label_col='exam_pathology_binary'):
    group_aucs = {}
    for group in df[col].unique():
        group_df = df[df[col] == group]
        y_true = group_df[label_col].values
        y_score = group_df[score_col].values
        if len(np.unique(y_true)) > 1:  # Ensure both classes are present
            group_aucs[group] = roc_auc_score(y_true, y_score)
        else:
            group_aucs[group] = np.nan  # Cannot compute AUC
    return group_aucs


# Function to compute bootstrapped AUCs for each group
def bootstrap_group_aucs(df, col, score_col='score', label_col='exam_pathology_binary', n_bootstraps=5000):
    group_bootstrapped_aucs = {}

    for group in df[col].unique():
        group_df = df[df[col] == group]
        y_true = group_df[label_col].values
        y_score = group_df[score_col].values

        if len(np.unique(y_true)) > 1:  # Ensure both classes are present
            bootstrapped_aucs = []
            for _ in range(n_bootstraps):
                indices = resample(range(len(y_true)))
                y_true_boot = y_true[indices]
                y_score_boot = y_score[indices]
                if len(np.unique(y_true_boot)) > 1:
                    bootstrapped_aucs.append(roc_auc_score(y_true_boot, y_score_boot))
            group_bootstrapped_aucs[group] = bootstrapped_aucs
        else:
            group_bootstrapped_aucs[group] = []  # Cannot compute AUC
    
    return group_bootstrapped_aucs


# Function to perform Kruskal-Wallis test on bootstrapped AUCs
def kruskal_test_on_aucs(bootstrapped_aucs):
    auc_lists = [bootstrapped_aucs[group] for group in bootstrapped_aucs if len(bootstrapped_aucs[group]) > 0]
    if len(auc_lists) > 1:  # Ensure there are at least two groups
        return kruskal(*auc_lists)
    return None


# Function to perform one-vs-each bootstrapped AUC difference tests
def perform_one_vs_each_auc_tests(bootstrapped_aucs, n_bootstraps=5000, ci=0.95):
    one_vs_each_results = []

    # Generate unique combinations of groups
    group_pairs = list(combinations(bootstrapped_aucs.keys(), 2))

    for group, other_group in group_pairs:
        if len(bootstrapped_aucs[group]) == 0 or len(bootstrapped_aucs[other_group]) == 0:
            continue  # Skip if either group has no valid AUCs

        auc_differences = np.array([
            auc_1 - auc_2
            for auc_1, auc_2 in zip(
                resample(bootstrapped_aucs[group], n_samples=n_bootstraps),
                resample(bootstrapped_aucs[other_group], n_samples=n_bootstraps)
            )
        ])

        # Confidence intervals
        auc_diff_mean = np.mean(auc_differences)
        ci_lower = np.percentile(auc_differences, (1 - ci) / 2 * 100)
        ci_upper = np.percentile(auc_differences, (1 + ci) / 2 * 100)

        # P-value (two-sided)
        p_value = 2 * min(np.mean(auc_differences > 0), np.mean(auc_differences < 0))

        # Append results
        one_vs_each_results.append({
            "Group": group,
            "Compared To": other_group,
            "AUC Difference Mean": auc_diff_mean,
            "95% CI": (ci_lower, ci_upper),
            "P Value": p_value
        })

    return pd.DataFrame(one_vs_each_results)


# Pairwise Permutation Test on Bootstrapped AUCs
def pairwise_permutation_test(bootstrapped_aucs, n_permutations=5000):
    pairwise_results = []

    # Generate unique group pairs using combinations
    group_pairs = list(combinations(bootstrapped_aucs.keys(), 2))

    for group_1, group_2 in group_pairs:
        aucs_1 = bootstrapped_aucs[group_1]
        aucs_2 = bootstrapped_aucs[group_2]

        if len(aucs_1) == 0 or len(aucs_2) == 0:
            continue  # Skip if any group has no valid AUCs

        # Compute observed difference in AUCs
        observed_diff = np.mean(aucs_1) - np.mean(aucs_2)

        # Combine AUCs and perform permutation
        combined_aucs = np.concatenate([aucs_1, aucs_2])
        labels = np.array([1] * len(aucs_1) + [2] * len(aucs_2))

        permuted_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(labels)
            permuted_aucs_1 = combined_aucs[labels == 1]
            permuted_aucs_2 = combined_aucs[labels == 2]
            permuted_diffs.append(np.mean(permuted_aucs_1) - np.mean(permuted_aucs_2))

        # Compute p-value
        permuted_diffs = np.array(permuted_diffs)
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

        # Store result
        pairwise_results.append({
            'Group 1': group_1,
            'Group 2': group_2,
            'Observed Difference': observed_diff,
            'P-Value': p_value
        })

    return pd.DataFrame(pairwise_results)


def apply_bonferroni_correction_groupwise(results_df, alpha=0.05):
    """
    Apply Bonferroni correction to a DataFrame of test results, adjusting p-values group-wise.

    Args:
        results_df (pd.DataFrame): DataFrame containing test results with 'Group Column' and 'P Value'.
        alpha (float): Original significance level (default is 0.05).

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted p-values and significance status.
    """
    # Add a column for adjusted p-values
    results_df['P Value Adjusted'] = 1.0  # Initialize adjusted p-value column

    # Group by the column defining groups (e.g., "Group Column")
    for group_col in results_df['Group Column'].unique():
        group_results = results_df[results_df['Group Column'] == group_col]

        # Number of pairwise comparisons for this group column
        num_tests = len(group_results)

        # Apply Bonferroni correction for this group
        results_df.loc[results_df['Group Column'] == group_col, 'P Value Adjusted'] = (
            group_results['P Value'] * num_tests
        ).clip(upper=1)  # Cap at 1 to ensure valid probabilities

    # Determine significance after correction
    results_df['Significant (Bonferroni)'] = results_df['P Value Adjusted'] < alpha

    return results_df


def bootstrap_partial_overlap_auc_test(df,
                                       invasive_label_col="Invasive\nCancer",
                                       noninvasive_label_col="DCIS",
                                       invasive_score_col="score_invasive",
                                       noninvasive_score_col="score_noninvasive",
                                       n_bootstraps=10000,
                                       ci=0.95):
    """
    Performs a bootstrap test to compare AUCs of two scenarios:
    - Invasive scenario: Invasive cancer (1) vs. No cancer (0)
    - Noninvasive scenario: Noninvasive cancer (1) vs. No cancer (0)
    
    Both share the same no cancer patients, but have distinct positive patient sets.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns for labels and scores.
        invasive_label_col (str): Column name for invasive scenario labels (1 = invasive, 0 = no cancer)
        noninvasive_label_col (str): Column name for noninvasive scenario labels (1 = noninvasive, 0 = no cancer)
        invasive_score_col (str): Column name for invasive scenario scores.
        noninvasive_score_col (str): Column name for noninvasive scenario scores.
        n_bootstraps (int): Number of bootstrap iterations.
        ci (float): Confidence interval level (e.g., 0.95 for 95% CI).
        
    Returns:
        diff (float): Observed difference in AUC (invasive_auc - noninvasive_auc).
        (ci_lower, ci_upper) (tuple): Confidence interval for the difference.
        p_value (float): P-value (two-sided) for the difference.
    """
    # Separate the data into three groups:
    # 1. Invasive positives
    invasive_positives = df[df["exam_pathology"] == invasive_label_col]
    # 2. Noninvasive positives
    noninvasive_positives = df[df["exam_pathology"] == noninvasive_label_col]
    # 3. Shared negatives (patients with no cancer)
    #    For a patient to be considered 'no cancer' for both scenarios,
    #    they should be labeled 0 in both invasive and noninvasive columns.
    negatives = df[~((df["exam_pathology"] == invasive_label_col) | (df["exam_pathology"] == noninvasive_label_col))]

    # Compute original AUCs
    # Invasive scenario AUC
    y_true_invasive = np.concatenate([
        invasive_positives['exam_pathology_binary'].values,
        negatives['exam_pathology_binary'].values
    ])
    y_score_invasive = np.concatenate([
        invasive_positives['score'].values,
        negatives['score'].values
    ])
    auc_invasive = roc_auc_score(y_true_invasive, y_score_invasive)

    # Noninvasive scenario AUC
    y_true_noninvasive = np.concatenate([
        noninvasive_positives['exam_pathology_binary'].values,
        negatives['exam_pathology_binary'].values
    ])
    y_score_noninvasive = np.concatenate([
        noninvasive_positives['score'].values,
        negatives['score'].values
    ])
    auc_noninvasive = roc_auc_score(y_true_noninvasive, y_score_noninvasive)

    observed_diff = auc_invasive - auc_noninvasive

    # Bootstrap differences
    diffs = []
    pos_invasive_size = len(invasive_positives)
    pos_noninvasive_size = len(noninvasive_positives)
    neg_size = len(negatives)

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping", unit="iter"):
        # Resample positives and negatives
        invasive_pos_boot = resample(invasive_positives, n_samples=pos_invasive_size, replace=True)
        noninvasive_pos_boot = resample(noninvasive_positives, n_samples=pos_noninvasive_size, replace=True)
        neg_boot = resample(negatives, n_samples=neg_size, replace=True)

        # Construct scenario arrays for invasive scenario
        y_true_invasive_boot = np.concatenate([
            invasive_pos_boot['exam_pathology_binary'].values,
            neg_boot['exam_pathology_binary'].values
        ])
        y_score_invasive_boot = np.concatenate([
            invasive_pos_boot['score'].values,
            neg_boot['score'].values
        ])

        # Construct scenario arrays for noninvasive scenario
        y_true_noninvasive_boot = np.concatenate([
            noninvasive_pos_boot['exam_pathology_binary'].values,
            neg_boot['exam_pathology_binary'].values
        ])
        y_score_noninvasive_boot = np.concatenate([
            noninvasive_pos_boot['score'].values,
            neg_boot['score'].values
        ])

        # Compute AUCs
        try:
            auc_inv_boot = roc_auc_score(y_true_invasive_boot, y_score_invasive_boot)
            auc_noninv_boot = roc_auc_score(y_true_noninvasive_boot, y_score_noninvasive_boot)
            diffs.append(auc_inv_boot - auc_noninv_boot)
        except ValueError:
            # This can happen if one of the bootstrap samples ends up with only one class.
            # In that case, skip this iteration.
            print("Error")
            continue

    diffs = np.array(diffs)

    # Compute p-value
    # Two-sided p-value = 2 * min(P(diff_boot >= observed_diff), P(diff_boot <= observed_diff))
    # p_val = 2 * min((diffs >= observed_diff).mean(), (diffs <= observed_diff).mean())
    p_val = 2 * min(np.mean(diffs > 0), np.mean(diffs < 0))

    # Confidence interval
    alpha = 1 - ci
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    return observed_diff, (ci_lower, ci_upper), p_val