import numpy as np, pandas as pd, pickle, json, re

from datetime import datetime
from json import JSONEncoder
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

seed = 88
np.random.seed(seed)

def get_minibatches_idx(n, batch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start : minibatch_start + batch_size])
        minibatch_start += batch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

def split_indices(nTotal, ratio, shuffle=False):
    """split index by ratio"""
    assert type(ratio) in (tuple, list), "type of ratio must in (tuple, list)"

    lenAry = [int(nTotal * e) for e in ratio]
    offset = 0
    peice = []
    ary = np.arange(0, nTotal)
    if shuffle:
        ary = np.random.permutation(ary)
        
    for i, len_ in enumerate(lenAry):
        if i < len(lenAry) - 1:
            peice.append(ary[offset : offset + len_])
        # last round
        else:
            peice.append(ary[offset:])
        offset += len_
    return peice

def preview(fpath, chunksize=5, names=None):
    for chunk in pd.read_csv(fpath, chunksize=chunksize, names=names):
        return chunk

def load_pickle(fpath):
    with open(fpath, "rb") as r:
        return pickle.load(r)

def dump_pickle(fpath, obj):
    with open(fpath, "wb") as w:
        pickle.dump(obj, w)


from collections import Counter
def split_ratings(data, pos_thres=4, testRatio=0.3):
    """依照比例切割train test資料"""
    tr, te = [], []
    for u, df in data.groupby("userId"):
        if len(df) < 5: continue

        pos, neg = df.query("rating >= {}".format(pos_thres)), df.query("rating < {}".format(pos_thres))
        pos_len = int(len(pos) * (1 - testRatio))
        tr_pos = pos[:pos_len]
        te_pos = pos[pos_len:]

        neg_len = int(len(neg) * (1 - testRatio))
        tr_neg = neg[:neg_len]
        te_neg = neg[neg_len:]

        tr.append(tr_pos.append(tr_neg))
        te.append(te_pos.append(te_neg))
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)

def doMovies(movies):
    """處理 movie: genres 轉換成數字"""
    movies = movies.reset_index(drop=True)
    movies.loc[movies.genres == "(no genres listed)", "genres"] = ""
    movies["genres"] = movies.genres.str.split("\|")
    genresMap = Counter()
    movies.genres.map(genresMap.update)
    om = CatgMapper().fit([e[0] for e in genresMap.most_common()])
    movies["genres"] = movies.genres.map(lambda lst: [om.enc[e] for e in lst])
    return movies, om


def auc_mean(y, pred_mat):
    """mean auc score of each user => shape[0] of y"""
    tot_auc, cnt = 0, 0
    for i in range(len(y)):
        nnz = y[i].nonzero()[0]
        if len(nnz) <= 1: continue

        labels = y[i][nnz]
        labels = (labels >= 4).astype(int)
        pred = pred_mat[i][nnz]
        if (labels == 1).all() or (labels == 0).all(): continue

        # print(i, ":", labels, predProba[i][nnz])
        tot_auc += roc_auc_score(labels, pred)
        cnt += 1
    return tot_auc / cnt


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def precision_at_k(truth, pred_mat, k=10, label_thres=4, pred_thres=0.8):
    hits, total = 0, 0
    for labels, pr in zip(truth, pred_mat):
        nnz = labels.nonzero()[0]
        if sum(labels >= label_thres) < k:
            continue


        # top_percentile = np.percentile(pr, pred_thres * 100)
        top_k_ind = (pr * (labels > 0)).argsort()[::-1][:k]
        hits += sum(labels[top_k_ind] >= label_thres)
        total += k
    return hits / total


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


from sklearn.base import BaseEstimator, TransformerMixin


class BaseMapper(BaseEstimator, TransformerMixin):
    def fit(self, y):
        return self.partial_fit(y)

    def partial_fit(self, y):
        return self

    def transform(self, y):
        return pd.Series(y).map(self.enc).fillna(0).values

    def fit_transform(self, y, **fit_params):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return pd.Series(y).map(self.inv_enc).values

    @staticmethod
    def from_json(cls, json_str):
        return cls().from_json(json_str)


class CounterEncoder(BaseMapper):
    """依照出現頻率進行編碼, 頻率由高到低的index = 0, 1, 2, 3 ..., 以此類推"""
    def __init__(self, n_total:int):
        self.counter = Counter()
        self.classes_ = []

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            for _ in y: break
            self.counter.update(y)
        except:
            self.counter.update([y])

        pair = self.counter.most_common()
        self.classes_ = list(pair[:, 0])

        idx = pair[:, 1]
        self.enc = dict(zip(self.classes_, idx))
        self.inv_enc = dict(zip(idx, self.classes_))
        return self

class CatgMapper(BaseMapper):
    """fit categorical feature"""
    def __init__(self, name=None, padding_null=False, is_multi=False, sep=None):
        self.name = name
        self.exists_ = set() if not padding_null else set([None])
        self.classes_ = []  if not padding_null else [None]
        self.is_multi = is_multi
        self.sep = sep
        self.enc = None
        self.inv_enc = None
        # self.n_total = n_total

    def init_check(self):
        return self

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            for _ in y: break
        except Exception as e:
            y = [y]

        # n_next = len(self.classes_) + len(batch)
        # if n_next > self.n_total:
        #     raise Exception('number of unique value out of range, '
        #                     'expected {}, got {}'.format(self.n_total, n_next))

        if self.is_multi:
            stack = set()
            y = pd.Series(list(set(y)))
            y.str.split('\s*{}\s*'.format(re.escape(self.sep))) \
                 .map(lambda e: stack.update(e if e is not None else []))
            y = stack
        else:
            y = set(y)

        batch = list(y - self.exists_)
        if len(batch):
            self.classes_ += batch
            self.exists_.update(self.classes_)

            idx = list(range(len(self.classes_)))
            self.enc = dict(zip(self.classes_, idx))
            self.inv_enc = dict(zip(idx, self.classes_))
        return self

    def transform(self, y):
        if self.is_multi:
            pd.Series(y).str.split('\s*{}\s*'.format(re.escape(self.sep))) \
                            .map(lambda ary: ','.join([self.enc[e] for e in ary]))
        else:
            return pd.Series(y).map(self.enc).values

    def to_json(self):
        info = {
            'name': self.name,
            'classes_': self.classes_,
            'is_multi': self.is_multi,
            'sep': self.sep
        }
        return json.dumps(info)

    def from_json(self, json_str):
        info = json.loads(json_str)
        self.classes_ = info['classes_']
        self.exists = set(self.classes_)
        idx = list(range(len(self.classes_)))
        self.enc = dict(zip(self.classes_, idx))
        self.inv_enc = dict(zip(idx, self.classes_))
        self.is_multi = info['is_multi']
        self.sep = info['sep']
        self.name = info['name']
        return self


class NumericMapper(BaseMapper):
    """fit numerical feature"""
    def __init__(self, name=None):
        self.name = name
        self.scaler = MinMaxScaler()
        self.max_ = None
        self.min_ = None
        self.cumsum_ = 0
        self.n_total_ = 0

    @property
    def mean(self):
        return self.cumsum_ / self.n_total_ if self.n_total_ > 0 else None

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            y = list(y)
        except ValueError as e:
            y = list([y])

        assert not isinstance(y[0], str), 'NumericMapper requires numeric data, got string!'

        y = pd.Series(y).dropna().values
        if len(y):
            self.cumsum_ += sum(y)
            self.n_total_ += len(y)
            self.scaler.partial_fit([[min(y)], [max(y)]])
            self.max_ = self.scaler.data_max_[0]
            self.min_ = self.scaler.data_min_[0]
            # self.scaler.partial_fit(y[:, np.newaxis])
        return self

    def transform(self, y):
        y = pd.Series(y).fillna(self.mean)[:, np.newaxis]
        return self.scaler.transform(y).reshape([-1])

    def inverse_transform(self, y):
        y = np.array(y)[:, np.newaxis]
        return self.scaler.inverse_transform(y).reshape([-1])

    def _to_json(self):
        return {
            'name': self.name,
            'max_': self.max_,
            'min_': self.min_,
            'cumsum_': self.cumsum_,
            'n_total_': self.n_total_
        }

    def _from_json(self, info):
        self.scaler = MinMaxScaler()
        self.max_ = info['max_']
        self.min_ = info['min_']
        self.scaler.partial_fit([[self.max_], [self.min_]])
        self.cumsum_ = info['cumsum_']
        self.n_total_ = info['n_total_']
        self.name = info['name']

    def to_json(self):
        return json.dumps(self._to_json())

    def from_json(self, json_str):
        info = json.loads(json_str)
        self._from_json(info)
        return self

class DatetimeMapper(NumericMapper):
    def __init__(self, name=None, dt_fmt=None):
        super().__init__(name)
        self.dt_fmt = dt_fmt

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            y = list(y)
        except ValueError as e:
            y = list([y])

        assert isinstance(y[0], str), 'DatetimeMapper requires string data for parsing, got {}!'.format(type(y[0]))

        y = pd.Series(y).dropna().map(lambda e: datetime.strptime(e, self.dt_fmt).timestamp()).values
        if len(y):
            self.cumsum_ += sum(y)
            self.n_total_ += len(y)
            self.scaler.partial_fit([[min(y)], [max(y)]])
            self.max_ = self.scaler.data_max_[0]
            self.min_ = self.scaler.data_min_[0]
            # self.scaler.partial_fit(y[:, np.newaxis])
        return self

    def transform(self, y):
        y = pd.Series(y).map(lambda e: datetime.strptime(e, self.dt_fmt).timestamp())
        y = y.fillna(self.mean)[:, np.newaxis]
        return self.scaler.transform(y).reshape([-1])

    def _to_json(self):
        info = super()._to_json()
        info['dt_fmt'] = self.dt_fmt
        return info

    def _from_json(self, info):
        super()._from_json(info)
        self.dt_fmt = info['dt_fmt']