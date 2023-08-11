import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN


def _generate_sample_smote(smote, x, x_tot, nnk, size=1):
    samples_indices = np.random.randint(low=0, high=nnk.size, size=1)
    rows = np.floor_divide(samples_indices, nnk.shape[1])
    cols = np.mod(samples_indices, nnk.shape[1])
    steps = np.random.uniform(size=size)[:, np.newaxis]
    sample = smote._generate_samples(x, x_tot, nnk, rows, cols, steps)
    return sample


def _generate_sample_adasyn(adasyn, x):
    adasyn.nn_.fit(x)
    nns = adasyn.nn_.kneighbors(x, return_distance=False)[:, 1:]
    rows = np.repeat(1, 1)
    cols = np.random.choice(5, size=1)
    diffs = x[nns[rows, cols]] - x[rows]
    steps = np.random.uniform(size=(1, 1))
    X_new = x[rows] + steps * diffs
    X_new = X_new.astype(x.dtype)
    return X_new


def _balance_set(w_exp, w_obs, df: pd.DataFrame, tot_df, strategy, round_level=None, debug=False, k=-1):
    disp = round(w_exp / w_obs, round_level) if round_level else w_exp / w_obs
    disparity = [disp]
    i = 0
    x_tot = tot_df.values

    if strategy == 'smote':
        smote = SMOTE()
        smote._validate_estimator()
        smote.nn_k_.fit(x_tot)
        nnk = smote.nn_k_.kneighbors(x, return_distance=False)[:, 1:]


    if strategy == 'adasyn':
        adasyn = ADASYN()
        adasyn._validate_estimator()

    while disp != 1 and i != k:
        x = df.values
        if w_exp / w_obs > 1:
            if strategy == 'smote':
                sample = _generate_sample_smote(smote, x, x_tot, nnk)
                df = df.append(pd.DataFrame(sample.reshape(
                    1, -1), columns=list(df)), ignore_index=True)
            elif strategy == 'adasyn':
                sample = _generate_sample_adasyn(adasyn, x)
                df = df.append(pd.DataFrame(sample.reshape(
                    1, -1), columns=list(df)), ignore_index=True)
            elif strategy == 'uniform':
                df = df.append(df.sample())
        elif w_exp / w_obs < 1:
            df = df.drop(df.sample().index, axis=0)
        w_obs = len(df) / len(tot_df)
        disp = round(
            w_exp / w_obs, round_level) if round_level else w_exp / w_obs
        disparity.append(disp)
        if debug:
            print(w_exp / w_obs)
        i += 1
    return df, disparity, i


def _sample(d: pd.DataFrame, s_vars: list, label: str, round_level: float, strategy: str, debug: bool = False,
            i: int = 0, G=None, cond: bool = True, stop=-1):
    if G is None:
        G = []
    d = d.copy()
    n = len(s_vars)
    disparities = []
    iter = 0
    if i == n:
        for l in np.unique(d[label]):
            g = d[(cond) & (d[label] == l)]
            if len(g) > 0:
                w_exp = (len(d[cond])/len(d)) * (len(d[d[label] == l])/len(d))
                w_obs = len(g)/len(d)
                g_new, disp, k = _balance_set(
                    w_exp, w_obs, g, d, strategy, round_level, debug, stop)
                g_new = g_new.astype(g.dtypes.to_dict())
                disparities.append(disp)
                G.append(g_new)
                iter = max(iter, k)
        return G, iter
    else:
        s = s_vars[i]
        i = i+1
        G1, k1 = _sample(d, s_vars, label, round_level, strategy, debug, i,
                         G.copy(), cond=cond & (d[s] == 0), stop=stop)
        G2, k2 = _sample(d, s_vars, label, round_level, strategy, debug, i,
                         G.copy(), cond=cond & (d[s] == 1), stop=stop)
        G += G1
        G += G2
        iter = max([iter, k1, k2])
        limit = 1
        for s in s_vars:
            limit *= len(np.unique(d[s]))
        if len(G) == limit*len(np.unique(d[label])):
            return pd.DataFrame(G.pop().append([g for g in G]).sample(frac=1, random_state=2)), disparities, iter
        else:
            return G, iter


class DEMV:
    '''
    Debiaser for Multiple Variable

    Attributes
    ----------
    round_level : float
        Tolerance value to balance the sensitive groups
    debug : bool
        Prints w_exp/w_obs, useful for debugging
    stop : int
        Maximum number of balance iterations
    strategy: string
        Balancing strategy to use. Must be one of `smote`, `adasyn` and `uniform` (default is `uniform`)
    iter : int
        Maximum number of iterations
    
    Parameters
    ----------
    round_level : float
        Tolerance value to balance the sensitive groups
    debug : bool
        Prints w_exp/w_obs, useful for debugging
    stop : int
        Maximum number of balance iterations
    strategy: string
        Balancing strategy to use. Must be one of `smote`, `adasyn` and `uniform` (default is `uniform`)

    Methods
    -------
    fit_transform(dataset, protected_attrs, label_name)
        Returns the balanced dataset

    get_iters()
        Returns the maximum number of iterations

    '''

    def __init__(self, round_level=None, debug=False, stop=-1, strategy='uniform'):
        '''
        Parameters
        ----------
        round_level : float, optional
            Tolerance value to balance the sensitive groups (default is None)
        debug : bool, optional
            Prints w_exp/w_obs, useful for debugging (default is False)
        stop : int, optional
            Maximum number of balance iterations (default is -1)
        strategy: string, optional
            Balancing strategy to use. Must be one of `smote`, `adasyn` and `uniform` (default is `uniform`)
        '''
        assert (strategy == 'uniform' or strategy == 'smote' or strategy == 'adasyn'),"Invalid strategy in DEMV, must be one of uniform, smote or adasyn"
        
        self.round_level = round_level
        self.debug = debug
        self.stop = stop
        self.iter = 0
        self.strategy = strategy

    def fit_transform(self, dataset: pd.DataFrame, protected_attrs: list, label_name: str):
        '''
        Balances the dataset's sensitive groups

        Parameters
        ----------
        dataset : pandas.DataFrame
            Dataset to be balanced
        protected_attrs : list
            List of protected attribute names
        label_name : str
            Label name

        Returns
        -------
        pandas.DataFrame :
            Balanced dataset
        '''
        df_new, disparities, iter = _sample(dataset, protected_attrs,
                                            label_name, self.round_level, self.strategy, self.debug, 0, [], True,
                                            self.stop)
        self.iter = iter
        return df_new

    def get_iters(self):
        '''
        Gets the maximum number of iterations

        Returns
        -------
        int:
            maximum number of iterations
        '''
        return self.iter