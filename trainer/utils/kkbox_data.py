import pandas as pd, os, pickle, numpy as np, time

from abc import abstractmethod
from collections import defaultdict, Counter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, LabelBinarizer
from .utils import splitIndices

np.random.seed(42)

def norm(data, cols, scaler):
    """normolize指定的欄位"""
    if not len(cols): return data

    ret = scaler.transform(data[cols])
    for col, val in zip(cols, ret.T):
        data.loc[:, col] = val
    return data

class DataHandler(object):
    def __init__(self):
        self.encMap = defaultdict(dict)
        self.oneHotMap = defaultdict(LabelBinarizer)
        self.embMap = {}
        self.scaler = MinMaxScaler()

    def clean(self, data):
        return data

    def fit(self, data):
        return self

    def transform(self, data):
        return data

    def fit_transform(self, data):
        cleanedData = self.clean(data)
        return self.fit(cleanedData).transform(cleanedData)

class MembersHandler(DataHandler):
    def clean(self, data):
        data.loc[(data.bd >= 80) | (data.bd <= 0), "bd"] = 0
        # gender outlier
        # data.loc[pd.isnull(data.gender), "gender"] = ""
        return data

    def fit(self, data):
        self.catgCols = ["city", "gender", "registered_via"]
        self.contCols = ["bd", "registration_init_time", "expiration_date"]
        self.oneHotCols = ["city", "gender", "registered_via"]
        self.embCols = ["msno"]

        for col in self.catgCols:
            series = data[col].dropna()
            self.encMap[col] = dict(zip([None] + list(series.value_counts().index), range(series.size + 1)))
            # self.encMap[col].update({None: 0})

        for col in self.embCols:
            series = data[col].dropna()
            self.embMap[col] = dict(zip([None] + list(series.unique()), range(series.size + 1)))
            # self.embMap[col].update({None: 0})

        # for col in self.oneHotCols:
        #     self.oneHotMap[col].fit(data[col])

        if len(self.contCols):
            self.scaler.fit(data[self.contCols])
        return self

    def transform(self, data):
        data = data.reset_index(drop=True)
        for col in self.catgCols:
            data.loc[:, col] = data[col].map(self.encMap[col])

        for col in self.embCols:
            data.loc[:, col] = data[col].map(self.embMap[col])

        data = norm(data, self.contCols, self.scaler)
        return data

class SongsHandler(DataHandler):
    def clean(self, data):
        # hashFn = lambda e: str(hash(tuple(set(e))))
        # data["genre_ids"] = data.genre_ids.fillna("")
        # data["artist_name"] = data.artist_name.fillna("")
        # data["composer"] = data.composer.fillna("")
        # data["lyricist"] = data.lyricist.fillna("")
        data["language"] = data.language.map(lambda e: e if pd.isnull(e) else 0 if e < 0 else int(e))
        return data

    def fit(self, data):
        self.catgCols = ["language"]
        self.contCols = ["song_length"]
        self.oneHotCols = ["language"]
        self.embCols = ["song_id", "genre_ids", "artist_name", "composer", "lyricist"]
        self.embMap = {}

        for col in self.catgCols:
            series = data[col].dropna()
            self.encMap[col] = dict(zip([None] + list(series.value_counts().index), range(series.size + 1)))

        for col in self.embCols:
            series = data[col].dropna()
            if col != "song_id":
                self.embMap[col] = Counter()
                series.map(lambda e: self.embMap[col].update( map(str.strip, e.strip().split("|")) ))
                self.embMap[col] = dict(zip([None] + np.array(self.embMap[col].most_common())[:, 0].tolist(), range(len(self.embMap[col]) + 1)))
                # self.embMap[col] = pd.Series(index=np.array(self.embMap[col].most_common())[:, 0], data=range(1, len(self.embMap[col]) + 1))
            else:
                self.embMap[col] = dict(zip([None] + list(series.unique()), range(series.size + 1)))

        # for col in self.oneHotCols:#
        #     self.oneHotMap[col].fit(data[col])

        if len(self.contCols):
            self.scaler.fit(data[self.contCols])
        return self

    def transform(self, data):
        data = data.copy()
        for col in self.catgCols:
            data.loc[:, col] = data[col].map(self.encMap[col])

        for col in self.embCols:
            if col != "song_id":
                data.loc[:, col] = data[col].map(lambda e: [0] if pd.isnull(e) else map(lambda o: self.embMap[col][o.strip()], e.strip().split("|")))
                                            # .map(lambda e: self.embMap[col][e].tolist())
            else:
                data.loc[:, col] = data[col].map(self.embMap[col])
        data = norm(data, self.contCols, self.scaler)
        return data

class BaseHandler(DataHandler):
    def clean(self, data):
        return data

    def fit(self, data):
        self.catgCols = ["source_system_tab", "source_screen_name", "source_type"]
        self.contCols = []
        self.oneHotCols = ["source_system_tab", "source_screen_name", "source_type"]
        self.embCols = ["msno", "song_id"]

        for col in self.catgCols:
            series = data[col].dropna()
            self.encMap[col] = dict(zip([None] + list(series.value_counts().index), range(series.size + 1)))
        # for col in self.oneHotCols:
        #     self.oneHotMap[col].fit(data[col])

        if len(self.contCols):
            self.scaler.fit(data[self.contCols])
        return self

    def transform(self, data):
        data = data.copy()
        for col in self.catgCols:
            data.loc[:, col] = data[col].map(self.encMap[col])

        for col in self.embCols:
            data.loc[:, col] = data[col].map(self.embMap[col])
        data = norm(data, self.contCols, self.scaler)
        return data

def preprocess(dataCtx):
    # if miss trainging file, do all preprocessing again
    if not os.path.isfile("{}/kkbox/kkbox.tr.csv".format(dataCtx)): doPreprocess(dataCtx)

    with open("{}/kkbox/kkboxDhMap.h".format(dataCtx), "rb") as r:
        dhMap = pickle.load(r)

    columns = ['target', 'msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type',
               'city', 'bd', 'gender', 'registered_via', 'registration_init_time', 'expiration_date',
               'song_length', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
    return pd.read_csv("{}/kkbox/kkbox.tr.csv".format(dataCtx),  encoding="utf-8", names=columns).fillna(""),\
            pd.read_csv("{}/kkbox/kkbox.vl.csv".format(dataCtx), encoding="utf-8", names=columns).fillna(""),\
            pd.read_csv("{}/kkbox/kkbox.te.csv".format(dataCtx), encoding="utf-8", names=columns).fillna(""), \
            dhMap

    # dtype = {catgCol: str for catgCol in ["msno", "song_id", "source_system_tab", "source_screen_name", "source_type",
    #                          "city", "gender", "registered_via", "genre_ids", "artist_name",
    #                          "composer", "lyricist", "language"]}
    # return pd.read_csv("{}/kkbox/kkbox.tr.csv".format(dataCtx),  names=columns, encoding="utf-8").fillna(""),\
    #         pd.read_csv("{}/kkbox/kkbox.vl.csv".format(dataCtx), names=columns, encoding="utf-8").fillna(""),\
    #         pd.read_csv("{}/kkbox/kkbox.te.csv".format(dataCtx), names=columns, encoding="utf-8").fillna(""), \
    #         dhMap


def doPreprocess(dataCtx):
    print("members ...")
    members = pd.read_csv("{}/kkbox/members.csv".format(dataCtx))
    membersDh = MembersHandler()
    members = membersDh.clean(members)
    membersDh.fit(members)

    print("songs ...")
    songs = pd.read_csv("{}/kkbox/songs.csv".format(dataCtx))
    songsDh = SongsHandler()
    songs = songsDh.clean(songs)
    songsDh.fit(songs)

    print("base ...")
    baseDh = BaseHandler()
    baseDh.encMap.update(membersDh.encMap)
    baseDh.embMap.update(membersDh.embMap)
    baseDh.oneHotMap.update(membersDh.oneHotMap)
    baseDh.encMap.update(songsDh.encMap)
    baseDh.embMap.update(songsDh.embMap)
    baseDh.oneHotMap.update(songsDh.oneHotMap)

    # 原始train data是百萬筆等級, 這裡隨機抽出150000筆
    train = pd.read_csv("{}/kkbox/train.csv".format(dataCtx)).dropna(how="any").sample(150000)
    train = train[train.song_id.isin(songs.song_id) & train.msno.isin(members.msno)].reset_index(drop=True)
    baseDh.fit(baseDh.clean(train))

    base = train.merge(members, on="msno", how="left").merge(songs, on="song_id", how="left")
    # move target column to first
    labelCol = "target"
    base = pd.concat([base[labelCol], base.drop(labelCol, 1)], axis=1)
    del train, songs, members

    # split data, 確保traing data每個user有15筆以上的資料, valid, test data的user都存在於在train data
    print("split train, valid, test data ...")
    trIdx, vlIdx, teIdx = splitIndices(len(base), (.7, .1, .2), shuffle=True)
    tr, vl, te = base.iloc[trIdx], base.iloc[vlIdx], base.iloc[teIdx]
    del base

    msnoCnt = tr.msno.value_counts()
    tr = tr[tr.msno.isin(msnoCnt[msnoCnt >= 15].index)]
    vl = vl[vl.msno.isin(tr.msno)].copy()
    te = te[te.msno.isin(tr.msno)].copy()

    tr.reset_index(drop=True).to_csv("{}/kkbox/kkbox.tr.raw.csv".format(dataCtx), header=None, index=False, encoding="utf-8")
    vl.reset_index(drop=True).to_csv("{}/kkbox/kkbox.vl.raw.csv".format(dataCtx), header=None, index=False, encoding="utf-8")
    te.reset_index(drop=True).to_csv("{}/kkbox/kkbox.te.raw.csv".format(dataCtx), header=None, index=False, encoding="utf-8")

    print("persistence ...")
    dhMap = {"membersDh": membersDh, "songsDh": songsDh, "baseDh": baseDh}
    with open("{}/kkbox/kkboxDhMap.h".format(dataCtx), "wb") as w:
        pickle.dump(dhMap, w)

    for df, topath in ((tr, '{}/kkbox/kkbox.tr.csv'.format(dataCtx)),\
                       (vl, '{}/kkbox/kkbox.vl.csv'.format(dataCtx)),\
                       (te, '{}/kkbox/kkbox.te.csv'.format(dataCtx))):
        persistence(df, topath, dhMap, nBatch=10000)


def persistence(fpath, topath, dhMap, nBatch=10000):
    trans = None
    for ftrs, label in dataFn(fpath, nBatch, dhMap):
        trans = pd.concat([trans, pd.concat([label, ftrs], 1)], ignore_index=True)
    trans.to_csv(topath, index=False, header=None, encoding="utf-8")

def dataFn(df, nBatch=10000, dhMap=None):
    baseDh, membersDh, songsDh = dhMap["baseDh"], dhMap["membersDh"], dhMap["songsDh"]

    # for batch in pd.read_csv(fpath, chunksize=nBatch, names=columns):
    # for pos in range(0, len(df), nBatch):
    for _, batch in df.groupby(np.arange(len(df)) // nBatch):
        # batch = df[pos:pos + nBatch].reset_index(drop=True)
        batch = batch.reset_index(drop=True)
        batch.loc[:, "msno"] = batch["msno"].map(baseDh.embMap["msno"])
        batch.loc[:, "song_id"] = batch["song_id"].map(baseDh.embMap["song_id"])
        batch.loc[:, "source_system_tab"] = batch["source_system_tab"].map(baseDh.encMap["source_system_tab"])
        batch.loc[:, "source_screen_name"] = batch["source_screen_name"].map(baseDh.encMap["source_screen_name"])
        batch.loc[:, "source_type"] = batch["source_type"].map(baseDh.encMap["source_type"])
        # members
        batch.loc[:, "city"] = batch["city"].map(membersDh.encMap["city"])
        batch.loc[:, "gender"] = batch["gender"].map(membersDh.encMap["gender"])
        batch.loc[:, "registered_via"] = batch["registered_via"].map(membersDh.encMap["registered_via"])
        batch = norm(data=batch,
                     cols=["bd", "registration_init_time", "expiration_date"],
                     scaler=membersDh.scaler)
        # songs
        batch.loc[:, "language"] = batch["language"].map(songsDh.encMap["language"])
        fn = lambda e: [0] if pd.isnull(e) else list(map(lambda o: songsDh.embMap[col][o.strip()], e.strip().split("|")))
        for col in ["genre_ids", "artist_name", "composer", "lyricist"]:
            batch.loc[:, col] = batch[col].map(fn)

        batch = norm(data=batch, cols=["song_length"], scaler=songsDh.scaler)

        yield batch.drop("target", 1), batch[["target"]]


