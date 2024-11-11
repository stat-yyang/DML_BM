import numpy as np

class LinearScoreMixin:
    """Mixin class implementing DML estimation for score functions being linear in the target parameter

    Notes
    -----
    The score functions of many DML models (PLR, PLIV, IRM, IIVM) are linear in the parameter :math:`\\theta`, i.e.,

    .. math::

        \\psi(W; \\theta, \\eta) = \\theta \\psi_a(W; \\eta) + \\psi_b(W; \\eta).

    """
    _score_type = 'linear'

    @property
    def _score_element_names(self):
        return ['psi_a', 'psi_b']

    def _compute_score(self, psi_elements, coef):
        psi = np.einsum('ijk,k->ij', psi_elements['psi_a'], coef) + psi_elements['psi_b']
        return psi

    def _compute_score_deriv(self, psi_elements, coef):
        return psi_elements['psi_a']

    def _est_coef(self, psi_elements, smpls=None, algorithm="DML2"):
        psi_a = psi_elements['psi_a']
        psi_b = psi_elements['psi_b']
        print(psi_a.shape)
        print(psi_b.shape)       
        if algorithm == "DML2":
            # Compute the inverse of the mean of psi_a across all observations
            j_hat = np.mean(psi_a, axis=0)
            print(j_hat.shape)
            j_hat_inv = np.linalg.inv(j_hat) if j_hat.ndim == 2 else 1.0 / j_hat
            coef = - np.dot(j_hat_inv, np.mean(psi_b, axis=0))
        elif algorithm == "DML1":
            coefs = []
            for idx, (train_index, test_index) in enumerate(smpls):
                psi_a_fold = psi_a[test_index]
                psi_b_fold = psi_b[test_index]
                j_hat_fold = np.mean(psi_a_fold, axis=0)
                j_hat_fold_inv = np.linalg.inv(j_hat_fold) if j_hat_fold.ndim == 2 else 1.0 / j_hat_fold
                coef_fold = - np.dot(j_hat_fold_inv, np.mean(psi_b_fold, axis=0))
                coefs.append(coef_fold)
            coef = np.mean(coefs, axis=0)
        else:
            raise ValueError('Invalid algorithm specified. Choose "DML1" or "DML2".')
        
        return coef

    def _est_sd(self, psi_elements, coef):
        psi_a = psi_elements['psi_a']
        psi_b = psi_elements['psi_b']

        score = self._compute_score(psi_elements, coef)

        print(score)
        # Estimate variance and standard error using pseudo-inverse for stability
        j_hat = np.mean(psi_a, axis=0)
        j_hat_inv = np.linalg.inv(j_hat) if j_hat.ndim == 2 else 1.0 / j_hat
        var_hat = np.dot(np.dot(j_hat_inv, np.mean(np.einsum('ij,ik->ijk', score, score), axis=0)), j_hat_inv.T)
        se = np.sqrt(np.diag(var_hat) / psi_b.shape[0])
        print(j_hat)
        print(var_hat)
        print(psi_b)
        return se
