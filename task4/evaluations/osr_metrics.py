import numpy as np
from sklearn.metrics import roc_auc_score


def calibrate_threshold(u_val, q=95.0):
    return float(np.percentile(u_val, q))


def acceptance_rate(u, tau):
    return float((u <= tau).mean())


def rejection_rate(u, tau):
    return float((u > tau).mean())


def auroc(u_known, u_unknown):
    
    y = np.r_[np.zeros(len(u_known)), np.ones(len(u_unknown))]
    return float(roc_auc_score(y, np.r_[u_known, u_unknown]))


def fpr_at_val_tau(u_unknown, tau):
   
    return acceptance_rate(u_unknown, tau)

def osr_summary(u_test, u_near, u_far, tau):
    """Every Table 1 / Table 2 number for one (model, score) pair at its frozen tau."""
    u_all = np.r_[u_near, u_far]
    return {
        'auroc_near': auroc(u_test, u_near),
        'auroc_far': auroc(u_test, u_far),
        'auroc_all': auroc(u_test, u_all),
        'test_acceptance': acceptance_rate(u_test, tau),
        'rej_near': rejection_rate(u_near, tau),
        'rej_far': rejection_rate(u_far, tau),
        'rej_all': rejection_rate(u_all, tau),
        'fpr95_near': fpr_at_val_tau(u_near, tau),
        'fpr95_far': fpr_at_val_tau(u_far, tau),
        'fpr95_all': fpr_at_val_tau(u_all, tau),
    }