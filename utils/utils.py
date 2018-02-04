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
        return pd.Series(y).map(self.enc_).fillna(0).values

    def fit_transform(self, y, **fit_params):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return pd.Series(y).map(self.inv_enc_).values

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
    def __init__(self, name=None, allow_null=True,
                 is_multi=False, sep=None,
                 vocab:list=None, vocab_path:str=None):
        self.name = name
        self.allow_null = allow_null
        self.exists_ = set()
        self.classes_ = []
        self.is_multi = is_multi
        self.sep = sep
        self.vocab = vocab
        self.vocab_path = vocab_path

        self.enc_ = None
        self.inv_enc_ = None
        self.init_check()

    def init_check(self):
        if self.vocab is not None and self.vocab_path is not None:
            raise ValueError("choose either vocab or vocab_path, can't specified both")
        return self

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            for _ in y: break
        except Exception as e:
            y = [y]

        y = pd.Series(list(set(y)))
        if not self.allow_null:
            assert not y.hasnans, '[{}]: null value detected'.format(self.name)

        y = y.dropna()
        if self.is_multi:
            stack = set()
            y.str.split('\s*{}\s*'.format(re.escape(self.sep))).map(stack.update)
            y = stack

        # batch = list(y - self.exists_)
        if len(y):
            clazz = set(self.classes_)
            clazz.update(y)
            self.classes_ = sorted(clazz)
            self.exists_.update(self.classes_)
            self.gen_mapper()
        return self

    def gen_mapper(self):
        """according classes_ to generate enc_(encoder) and inv_enc_(decoder)

        :return:
        """
        if self.allow_null:
            idx = [0] + list(range(1, len(self.classes_) + 1))
            val = [None] + self.classes_
        else:
            idx = list(range(0, len(self.classes_)))
            val = self.classes_

        self.enc_ = dict(zip(val, idx))
        self.inv_enc_ = dict(zip(idx, val))

    def transform(self, y):
        """transform data(must fit first)

        :param y: string or string list
        :return: string splited by comma sign
        """
        y = pd.Series(y)
        if not self.allow_null:
            assert not y.hasnans, '[{}]: null value detected'.format(self.name)

        if self.is_multi:
            def do_multi(ary):
                if ary is None or len(ary[0]) == 0:
                    return [0]
                return pd.Series(ary).map(self.enc_).fillna(0).astype(int).tolist()

            return y.str.split('\s*{}\s*'.format(re.escape(self.sep))) \
                    .map(do_multi).values
        else:
            return y.map(self.enc_).fillna(0).astype(int).values

    def to_json(self):
        info = {
            'name': self.name,
            'classes_': self.classes_,
            'is_multi': self.is_multi,
            'sep': self.sep,
            'allow_null': self.allow_null
        }
        return json.dumps(info)

    def from_json(self, json_str):
        info = json.loads(json_str)
        self.is_multi = info['is_multi']
        self.allow_null = info['allow_null']
        self.sep = info['sep']
        self.name = info['name']
        self.classes_ = info['classes_']
        self.exists_ = set(self.classes_)
        self.gen_mapper()
        return self


class NumericMapper(BaseMapper):
    """fit numerical feature"""
    def __init__(self, name=None, default=None):
        self.default = default
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
        return self

    def transform(self, y):
        y = pd.Series(y).fillna(self.default if self.default is not None else self.mean)[:, np.newaxis]
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
            'n_total_': self.n_total_,
            'default': self.default
        }

    def _from_json(self, info):
        self.scaler = MinMaxScaler()
        self.max_ = info['max_']
        self.min_ = info['min_']
        self.scaler.partial_fit([[self.max_], [self.min_]])
        self.cumsum_ = info['cumsum_']
        self.n_total_ = info['n_total_']
        self.name = info['name']
        self.default = info['default']

    def to_json(self):
        return json.dumps(self._to_json())

    def from_json(self, json_str):
        info = json.loads(json_str)
        self._from_json(info)
        return self

class DatetimeMapper(NumericMapper):
    def __init__(self, name=None, dt_fmt=None, default=None):
        super().__init__(name=name, default=default)
        self.dt_fmt = dt_fmt
        self.default_ = None
        if default is not None:
            assert isinstance(default, str), 'datetime default value must be string!'
            self.default = default.strip()
            try:
                self.default_ = datetime.strptime(self.default, self.dt_fmt).timestamp()
            except Exception as e:
                raise ValueError('parse default datetime [{}] failed!\n\n{}'.format(self.default, e))

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
        return self

    def transform(self, y):
        y = pd.Series(y).map(lambda e: datetime.strptime(e, self.dt_fmt).timestamp() if e is not None else None)\
                        .fillna(self.default_ if self.default_ is not None else self.mean)[:, np.newaxis]
        return self.scaler.transform(y).reshape([-1])

    def _to_json(self):
        info = super()._to_json()
        info['dt_fmt'] = self.dt_fmt
        info['default_'] = self.default_
        return info

    def _from_json(self, info):
        super()._from_json(info)
        self.dt_fmt = info['dt_fmt']
        self.default_ = info['default_']