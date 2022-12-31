from sklearn.metrics import confusion_matrix


def DifferenceStatisticalParity(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged,labels):
    """
    Calculates the statistical parity (or demographic parity) metric for a binary classification task.
        Args:
            y_pred: A list or array of predicted labels (Default: 0 or 1).
            y_true: A 2D pandas Dataframe of true labels and sensitive group labels (0 or 1).
            SensitiveCat: Sensitive Feature columns name of y_true Dataframe.
            outcome: Target columns name of y_true Dataframe.
            privileged: privileged value of the sensitive group (Default 1).
            unprivileged: unprivileged value of the sensitive group (Default 0).
            labels: a list of Target values (Default: [0, 1])
        Returns:
            statistical_parity_difference: The difference in The difference in the proportion of positive
            predictions between the two sensitive groups.
                """
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
    """
    Calculates the equal opportunity metric for a binary classification task.
        Args:
            y_pred: A list or array of predicted labels (Default: 0 or 1).
            y_true: A 2D pandas Dataframe of true labels and sensitive group labels (0 or 1).
            SensitiveCat: Sensitive Feature columns name of y_true Dataframe.
            outcome: Target columns name of y_true Dataframe.
            privileged: privileged value of the sensitive group (Default 1).
            unprivileged: unprivileged value of the sensitive group (Default 0).
            labels: a list of Target values (Default: [0, 1])
        Returns:
            equal_opportunity_difference: The difference in TPR between the two sensitive groups.
        """
    y_priv = y_pred[y_real[SensitiveCat]==privileged]
    y_real_priv = y_real[y_real[SensitiveCat]==privileged]
    y_unpriv = y_pred[y_real[SensitiveCat]==unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat]==unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome],y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv, labels=labels).ravel()

    return abs(TP_unpriv/y_real_unpriv.shape[0] - TP_priv/y_real_priv.shape[0])

def DifferenceAverageOdds(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged,labels):
    """
    Calculates the average odds metric for a binary classification task.
        Args:
            y_pred: A list or array of predicted labels (Default: 0 or 1).
            y_true: A 2D pandas Dataframe of true labels and sensitive group labels (0 or 1).
            SensitiveCat: Sensitive Feature columns name of y_true Dataframe.
            outcome: Target columns name of y_true Dataframe.
            privileged: privileged value of the sensitive group (Default 1).
            unprivileged: unprivileged value of the sensitive group (Default 0).
            labels: a list of Target values (Default: [0, 1])
        Returns:
            average_odds_difference: The difference in TPR and FPR between the two sensitive groups.
        """
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv,  labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,  labels=labels).ravel()
    return 0.5*(abs(FP_unpriv/y_real_unpriv.shape[0]-FP_priv/y_real_priv.shape[0])+abs(TP_unpriv/y_real_unpriv.shape[0]-TP_priv/y_real_priv.shape[0]))