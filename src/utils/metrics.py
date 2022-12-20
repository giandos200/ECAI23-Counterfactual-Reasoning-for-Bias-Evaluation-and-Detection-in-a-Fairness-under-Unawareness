from sklearn.metrics import confusion_matrix


def DifferenceStatisticalParity(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged,labels):
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    N_priv = y_real_priv.shape[0]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    N_unpriv = y_real_unpriv.shape[0]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,
                                                                  labels=labels).ravel()
    return abs((TP_priv+FP_priv)/N_priv - (TP_unpriv+FP_unpriv)/N_unpriv)

def DifferenceEqualOpportunity(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged, labels):
    # difference in True Positive rate
    y_priv = y_pred[y_real[SensitiveCat]==privileged]
    y_real_priv = y_real[y_real[SensitiveCat]==privileged]
    y_unpriv = y_pred[y_real[SensitiveCat]==unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat]==unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome],y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv, labels=labels).ravel()

    return abs(TP_unpriv/y_real_unpriv.shape[0] - TP_priv/y_real_priv.shape[0])

def DifferenceAverageOdds(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged,labels):
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv,  labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,  labels=labels).ravel()
    return 0.5*(abs(FP_unpriv/y_real_unpriv.shape[0]-FP_priv/y_real_priv.shape[0])+abs(TP_unpriv/y_real_unpriv.shape[0]-TP_priv/y_real_priv.shape[0]))