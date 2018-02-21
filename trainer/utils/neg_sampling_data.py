import numpy as np,  pandas as pd, os, codecs

randomSeed = 88

def doMovies(movies):
    """處理 movie genres 轉換成數字"""
    movies = movies.reset_index(drop=True)
    movies.loc[movies.genres == "(no genres listed)", "genres"] = ""
    movies["genres"] = movies.genres.str.split("\|")
    genresMap = Counter()
    movies.genres.map(genresMap.update)
    om = OrderedMapper().fit([e[0] for e in genresMap.most_common()])
    movies["genres"] = movies.genres.map(lambda lst: [om.enc[e] for e in lst])
    return movies, om

def randomMovieSamples(allIds, excludeIds, negSize):
    candidateMovieIds = set(allIds).difference(excludeIds)
    return np.random.RandomState(randomSeed).choice(sorted(candidateMovieIds), negSize, replace=False).tolist()


def doSingleUser(uid, df, movies, isTrain=True, lastK=200):
    df = df.sort_values("rating", ascending=False)
    size = len(df)
    if size < 5: return []
    if isTrain:
        df = df[:lastK]
        size = len(df)
    elif size >= 1000:
        return []

    genresFreq, genresRate = Counter(), defaultdict(float)
    df.genres.map(genresFreq.update)
    for _, r in df.iterrows():
        for e in r.genres:
            genresRate[e] += r.rating / (size - 1)

    queue = []
    for i, (_, r) in enumerate(df.query("rating >= 4").iterrows()):
        # eval的資料去除low rate的資料
        # if r.rating < 4: continue

        for e in r.genres:
            if genresFreq[e] == 1:
                del genresFreq[e]
                del genresRate[e]
            else:
                genresFreq[e] -= 1
                genresRate[e] -= r.rating / (size - 1)
        sortedGenresFreq = OrderedDict(sorted(genresFreq.items()))
        sortedGenresRate = OrderedDict(sorted(genresRate.items()))

        if isTrain:
            sampledMovieIds = []
        else:
            # random sample 1000 movie ids for evaludate
            sampledMovieIds = randomMovieSamples(movies.movieId, df.movieId, 1000)
        # columns: userId, queryMovieIds, queryRated, queryGenresIds, queryGenresFreq,
        #          queryGenresRated, sampleMovieIds, candidateGenresIds, candidateMovieId, candidateRated
        queue.append([
            uid,
            df.movieId[:i].tolist() + df.movieId[i + 1:].tolist(),
            df.rating[:i].tolist() + df.rating[i + 1:].tolist(),
            list(sortedGenresFreq.keys()),
            list(sortedGenresFreq.values()),
            list(sortedGenresRate.values()),
            sampledMovieIds,
            r.genres,
            r.movieId,
            r.rating
        ])
        for e in r.genres:
            genresFreq[e] += 1
            genresRate[e] += r.rating / (size - 1)

    # negative sampling for train data
    # if isTrain:
    #     negSize = size * 2
    #     candidateMovieIds = set(movies.movieId).difference(df.movieId)
    #     negMovieIds = np.random.RandomState(randomSeed).choice(sorted(candidateMovieIds), negSize,
    #                                                            replace=False).tolist()
    #     sortedGenresFreq = OrderedDict(sorted(genresFreq.items()))
    #     sortedGenresRate = OrderedDict(sorted(genresRate.items()))
    #     for negId in negMovieIds:
    #         movie_data = movies[movies.movieId == negId].iloc[0]
    #         # columns: userId, queryMovieIds, queryRated, queryGenresIds, queryGenresFreq,
    #         #          queryGenresRated, sampleMovieIds, candidateMovieId, candidateRated
    #         queue.append([
    #             uid,
    #             df.movieId.tolist(),
    #             df.rating.tolist(),
    #             list(sortedGenresFreq.keys()),
    #             list(sortedGenresFreq.values()),
    #             list(sortedGenresRate.values()),
    #             [],
    #             movie_data.genres,
    #             negId,
    #             0.
    #         ])
    return queue


def auc_mean(y, predProba):
    """mean auc score of each user"""
    tot_auc, cnt = 0, 0
    for i in range(len(y)):
        nnz = y[i].nonzero()[0]
        if len(nnz) <= 1: continue

        labels = y[i][nnz]
        labels = (labels >= 4).astype(int)
        pred = predProba[i][nnz]
        if (labels == 1).all() or (labels == 0).all(): continue

        # print(i, ":", labels, predProba[i][nnz])
        tot_auc += roc_auc_score(labels, pred)
        cnt += 1
    print("auc:", tot_auc / cnt)

def dataFn(data, movies, isTrain=True, lastK=200):
    """movielens preprocessing, 每個user產生正向與負向資料, 每一筆資料皆包含user過去的history
       排除當前的candidate movie 資料 => leave one out 原理."""
    data = data.merge(movies, how="left", on="movieId")
    uCnt = 0
    for uid, df in data.groupby("userId"):
        yield doSingleUser(uid, df, movies, isTrain=isTrain, lastK=lastK)
        uCnt += 1
        print("\r{} user processed ...".format(uCnt), end="")


def negSampling2Csv(data, fpath, movies, nBatch=2000, isTrain=True):
    moviesTrans, om = doMovies(movies)
    if os.path.exists(fpath): os.remove(fpath)

    queue = []
    with codecs.open(fpath, "a") as w:
        for i, rows in enumerate(dataFn(data, moviesTrans, isTrain=isTrain), 1):
            queue.extend(rows)
            if len(queue) >= nBatch:
                pd.DataFrame(queue).to_csv(w, index=False, header=None)
                queue = []
        if len(queue):
            pd.DataFrame(queue).to_csv(w, index=False, header=None)
            queue = []