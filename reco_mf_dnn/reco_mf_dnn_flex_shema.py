import numpy as np, tensorflow as tf, os, shutil, time, json, pandas as pd, codecs
import shutil

from collections import OrderedDict
from utils import utils

seed = 88
np.random.seed(seed)
tf.set_random_seed(seed)
pd.set_option('display.width', 100)

class Schema(object):
    # config attrs
    COLUMNS = "columns"
    USER = 'user'
    ITEM = 'item'
    LABEL = 'label'

    ID = 'id'
    DATE_FORMAT = 'date_format'
    M_DTYPE = 'm_dtype'
    # continous, datetime
    CONT = 'cont'
    DATETIME = 'datetime'
    # categorical
    CATG = 'catg'
    DEFAULT = 'default'
    IS_MULTI = 'is_multi'
    VOCAB = 'vocab'
    VOCAB_PATH = 'vocab_path'
    SEP = 'sep'
    AUX = 'aux'
    M_DTYPE_ARY = [CONT, CATG, DATETIME]
    TYPE = 'type'
    COL_STATE = 'col_state'


    COL_ATTR = [ID, M_DTYPE, DATE_FORMAT, DEFAULT, IS_MULTI, SEP, AUX, TYPE, COL_STATE]
    COL_TYPE = [np.str, np.str, np.str, np.str, np.bool, np.str, np.bool, np.str, np.str]
    DEFAULT  = ['', '', '', 0, False, '', False, '', '']

    def __init__(self, json_path, parsed_json_path, raw_paths:list, unserialize=False):
        """Schema configs

        :param json_path: path for json columns specs configurations
        :param parsed_json_path: path for parsed json configurations
        :param raw_paths: multiple raw training csv files
        :param unserialize:
        """
        self.json_path = json_path
        self.parsed_json_path = parsed_json_path
        # TODO: wait for fetch GCS training data to parse
        self.raw_paths = raw_paths

        self.count_ = 0
        self.conf_ = None
        self.df_conf_ = None
        self.col_states_ = None
        if not unserialize:
            self.extract(self.parse_json()).check().fit()

    def parse_json(self):
        """flexible way to fetch json string from local, GCS, etc.

        :return: json string
        """
        # TODO: wait for fetch gcs config json file and read ...
        import codecs
        with codecs.open(self.json_path, 'r', encoding='utf-8') as r:
            return r.read()

    def extract(self, json_conf):
        """extract JSON config file

        :param json_conf:
        :return:
        """
        self.conf_ = json.loads(json_conf)
        for k in (Schema.USER, Schema.ITEM, Schema.LABEL, Schema.COLUMNS):
            assert k in self.conf_, 'config requires {} attrs, actual {}'\
                .format([Schema.USER, Schema.ITEM, Schema.LABEL, Schema.COLUMNS], list(self.conf_.keys()))

        cols = []
        for r in self.conf_[Schema.COLUMNS]:
            cols.append(r)

        self.df_conf_ = (pd.DataFrame(columns=Schema.COL_ATTR, data=cols)
                           .reset_index(drop=True))

        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['item']), 'type'] = 'item'
        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['user']), 'type'] = 'user'
        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['label']), 'type'] = 'label'

        for col, tpe, default in zip(Schema.COL_ATTR, Schema.COL_TYPE, Schema.DEFAULT):
            self.df_conf_[col] = self.df_conf_[col].fillna(default).astype(tpe)

        return self

    def check(self):
        """check user input columns configs

        :return: self
        """

        # check if basic attr exists
        df_conf = self.df_conf_.query("type != ''")
        base = df_conf.query("{} == '' or {} == ''".format(Schema.ID, Schema.M_DTYPE))
        assert len(base) == 0, 'require {} attrs, check following settings:\n{}'\
            .format([Schema.ID, Schema.M_DTYPE], base)

        # check if user, item, label columns in self.conf_['columns']
        for k in (Schema.USER, Schema.ITEM, Schema.LABEL):
            unknowns = set(self.conf_[k]) - set(df_conf[Schema.ID])
            assert len(unknowns) == 0, '{} not found in {} column settings'.format(list(unknowns), k)

        # check if dtype in [str, float, int, datetime]
        # err_dtypes = conf[~conf[Schema.DTYPE].isin(Schema.DTYPE_ARY)]
        # assert len(err_dtypes) == 0, 'require value of {} in {}, check following:\n{}'\
        #     .format(Schema.DTYPE, Schema.DTYPE_ARY, err_dtypes)

        # check if m_dtype in [cont, catg, datetime]
        err_m_dtypes = df_conf[~df_conf[Schema.M_DTYPE].isin(Schema.M_DTYPE_ARY)]
        assert len(err_m_dtypes) == 0, 'require value of {} in {}, check following:\n{}' \
            .format(Schema.M_DTYPE, Schema.M_DTYPE_ARY, err_m_dtypes)

        # check catg columns
        self.check_catg(df_conf)

        # datetime column requires date_format settings
        dt_no_format = df_conf.query("{} == 'datetime' and date_format == ''".format(Schema.M_DTYPE))
        assert len(dt_no_format) == 0, 'datetime column expect {} attr, check following:\n{}' \
            .format(Schema.DATE_FORMAT, dt_no_format)
        return self

    def check_catg(self, df_conf):
        """check catg columns
        :param df_conf: config in pandas
        :return: None
        """
        catg = df_conf.query("{} == '{}'".format(Schema.M_DTYPE, Schema.CATG))
        # null_n_unique = catg.query("{} <= 0".format(Schema.N_UNIQUE))
        # assert len(null_n_unique) == 0, 'categorical column expect number of vocabs [{}] value > 0, ' \
        #                                 'check following:\n{}'.format(Schema.N_UNIQUE, null_n_unique)

        multi_no_sep = catg.query("{} == True and {} == ''".format(Schema.IS_MULTI, Schema.SEP))
        assert len(multi_no_sep) == 0, 'multivalent column expect {} attr, check following:\n{}' \
            .format(Schema.SEP, multi_no_sep)

    def raw_dtype(self, df_conf):
        # str dtype for all catg + datetime columns, float dtype for all cont columns
        catg = df_conf.query("{} == '{}'".format(Schema.M_DTYPE, Schema.CATG))
        dt = df_conf.query("{} == '{}'".format(Schema.M_DTYPE, Schema.DATETIME))
        cont = df_conf.query("{} == '{}'".format(Schema.M_DTYPE, Schema.CONT))
        dt_catg = pd.concat([dt, catg], ignore_index=True)

        dtype = dict(zip(dt_catg[Schema.ID], ['str'] * len(dt_catg)))
        dtype.update(dict(zip(cont[Schema.ID], ['float'] * len(cont))))
        return dtype

    def fit(self):
        """fetch columns states in training data

        :return:
        """
        from datetime import  datetime

        df_conf = self.df_conf_.query("{} != ''".format(Schema.TYPE))
        dtype = self.raw_dtype(df_conf)

        col_states = OrderedDict()
        # './merged_movielens.csv'
        for fpath in self.raw_paths:
            for chunk in pd.read_csv(fpath,
                                     names=df_conf[Schema.ID].values,
                                     chunksize=10000, dtype=dtype):

                chunk = chunk.where(pd.notnull(chunk), None)
                # loop all valid columns except label
                for _, r in df_conf.iterrows():
                    val, m_dtype, name, col_type = None, r[Schema.M_DTYPE], r[Schema.ID], r[Schema.TYPE]

                    if col_type == Schema.LABEL:
                        if name not in col_states:
                            col_states[name] = utils.CatgMapper(name, allow_null=False)
                        # null value is not allowed in label column
                        assert not chunk[name].hasnans, 'null value detected in label column! filename {}' \
                            .format(fpath)
                    else:
                        # categorical column
                        if m_dtype == Schema.CATG:
                            is_multi, sep = r[Schema.IS_MULTI], r[Schema.SEP]
                            if name not in col_states:
                                if is_multi:
                                    col_states[name] = utils.CatgMapper(name, is_multi=is_multi, sep=sep)
                                else:
                                    col_states[name] = utils.CatgMapper(name)
                        # numeric column
                        elif m_dtype == Schema.CONT:
                            if name not in col_states:
                                col_states[name] = utils.NumericMapper(name)
                        # datetime column: transform to numeric
                        elif m_dtype == Schema.DATETIME:
                            dt_fmt = r[Schema.DATE_FORMAT]
                            if name not in col_states:
                                col_states[name] = utils.DatetimeMapper(name, dt_fmt)

                    col_states[name].partial_fit(chunk[name].values)

                # count data size
                self.count_ += len(chunk)

        self.col_states_ = col_states
        valid_cond = self.df_conf_[Schema.TYPE] != ''
        self.df_conf_.loc[valid_cond, 'col_state'] = \
            self.df_conf_.loc[valid_cond, Schema.ID].map(lambda e: col_states[e].to_json())

        # serialize to specific path
        with codecs.open(self.parsed_json_path, 'w', 'utf-8') as w:
            w.write(self.to_json())

    def to_json(self):
        """schema serialize to json

        :return: json string
        """
        # ret = self.df_conf_.to_json(orient='records')
        return json.dumps({
            'json_path': self.json_path,
            'parsed_json_path': self.parsed_json_path,
            'raw_paths': self.raw_paths,
            'count_': self.count_,
            'conf_': self.conf_,
            'df_conf_': self.df_conf_.to_dict(orient='records'),
        })

    @staticmethod
    def from_json(json_str):
        """Create Schema instance from json string

        :param json_str:
        :return:
        """
        info = json.loads(json_str)
        json_path, tr_paths = info['json_path'], info['tr_paths']
        this = Schema(json_path, tr_paths, unserialize=True)
        for k, attr in info.items():
            setattr(this, k, attr)

        this.df_conf_ = pd.DataFrame(this.df_conf_, columns=Schema.COL_ATTR)
        # specific class for each type
        ser_maps = {Schema.CATG: utils.CatgMapper,
                    Schema.CONT: utils.NumericMapper,
                    Schema.DATETIME: utils.DatetimeMapper}

        this.col_states_ = OrderedDict()
        for _, r in this.df_conf_.query("type != ''").iterrows():
            this.col_states_[r[Schema.ID]] = \
                    utils.BaseMapper.from_json(ser_maps[r[Schema.M_DTYPE]], r[Schema.COL_STATE])
        return this

# class Conf(
#     collections.namedtuple("Conf",
#                            ("initializer", "source", "target_input",
#                             "target_output", "source_sequence_length",
#                             "target_sequence_length"))):
#     pass


class Loader(object):
    def __init__(self, json_path, parsed_json_path, raw_paths:list=None):
        self.json_path = json_path
        self.parsed_json_path = parsed_json_path
        self.raw_paths = raw_paths
        self.schema = None

    def load(self, data_path):
        # init schema
        if self.schema is None:
            # 1. try unserialize
            if os.path.isfile(self.parsed_json_path):
                with codecs.open(self.json_path, 'r', 'utf-8') as r:
                    Schema.from_json( json.load(r) )
            # 2. if parsed_json_path not exists, try re-parse raw json (json_path supplied by user)
            else:
                self.schema = Schema(self.json_path, self.raw_paths, self.parsed_json_path)

        if not os.path.isfile(data_path):
            print('try to generate training data ... ')
            self.gen_tr_vl(data_path)

    def gen_tr_vl(self, data_path):
        # delete original file
        shutil.rmtree(data_path, ignore_errors=True)

        df_conf = self.schema.df_conf_.set_index('id', drop=False)
        dtype = self.schema.raw_dtype(df_conf)
        columns = df_conf[Schema.ID].values
        col_states = self.schema.col_states_
        # 70% training data, 30% testing data
        rand_seq = np.random.random(size=self.schema.count_)

        with codecs.open(data_path, 'a', 'utf-8') as w:
            for fpath in self.schema.raw_paths:
                # hack 100 records test
                for chunk in pd.read_csv(fpath, names=columns, chunksize=100, dtype=dtype):
                    chunk = chunk.where(pd.notnull(chunk), None)

                    for colname, col in chunk.iteritems():
                        if not colname in col_states: continue

                        # multivalent categorical columns
                        if df_conf.loc[colname, Schema.M_DTYPE] == Schema.CATG and \
                           df_conf.loc[colname, Schema.IS_MULTI]:
                            val = pd.Series(col_states[colname].transform(col))
                            # because of persistence to csv, transfer int array to string splitted by comma
                            chunk[colname] = val.map(lambda ary: ','.join(map(str, ary))).tolist()
                        # univalent columns
                        else:
                            chunk[colname] = list(col_states[colname].transform(col))
                    chunk.to_csv(w, index=False, header=None)
                    break



class ModelMfDNN(object):
    def __init__(self,
                 n_items,
                 n_genres,
                 model_dir,
                 dim=16,
                 learning_rate=0.01,
                 callback=None):
        self.n_items = n_items
        self.n_genres = n_genres
        self.ftr_cols = OrderedDict()
        self.callback = callback

        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("inputs"):
                self.is_train = tf.placeholder(tf.bool, None)
                # user data
                self.query_movie_ids = tf.placeholder(tf.int32, [None, None])
                self.query_movie_ids_len = tf.placeholder(tf.int32, [None])

                # item data
                self.genres = tf.placeholder(tf.int32, [None, None])
                self.genres_len = tf.placeholder(tf.int32, [None])
                self.avg_rating = tf.placeholder(tf.float32, [None])
                self.year = tf.placeholder(tf.float32, [None])
                self.candidate_movie_id = tf.placeholder(tf.int32, [None])
                self.rating = tf.placeholder(tf.float32, [None])

            init_fn = tf.glorot_normal_initializer()
            emb_init_fn = tf.glorot_uniform_initializer()
            self.b_global = tf.Variable(emb_init_fn(shape=[]), name="b_global")
            with tf.variable_scope("embedding"):
                self.w_query_movie_ids = tf.Variable(emb_init_fn(shape=[self.n_items, dim]), name="w_query_movie_ids")
                self.b_query_movie_ids = tf.Variable(emb_init_fn(shape=[dim]), name="b_query_movie_ids")
                self.w_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items, dim]), name="w_candidate_movie_id")
                self.b_candidate_movie_id = tf.Variable(init_fn(shape=[dim + 8 + 2]), name="b_candidate_movie_id")
                # self.b_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items]), name="b_candidate_movie_id")
                self.w_genres = tf.Variable(emb_init_fn(shape=[self.n_genres, 8]), name="w_genres")

            with tf.variable_scope("user_encoding"):
                # query_movie embedding
                self.emb_query = tf.nn.embedding_lookup(self.w_query_movie_ids, self.query_movie_ids)
                query_movie_mask = tf.expand_dims(
                    tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.query_movie_ids_len)), 1), -1)
                self.emb_query = tf.reduce_sum(self.emb_query * query_movie_mask, 1)
                self.query_bias = tf.matmul(self.emb_query, self.b_query_movie_ids[:, tf.newaxis])
                self.emb_query = tf.layers.dense(self.emb_query, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_query = tf.layers.dense(self.emb_query, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_query = tf.layers.dense(self.emb_query, 32, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_query = tf.layers.dense(self.emb_query, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

            # encode [item embedding + item metadata]
            with tf.variable_scope("item_encoding"):
                # candidate_movie embedding
                self.candidate_emb = tf.nn.embedding_lookup(self.w_candidate_movie_id, self.candidate_movie_id)
                # genres embedding
                self.emb_genres = tf.nn.embedding_lookup(self.w_genres, tf.to_int32(self.genres))
                genres_mask = tf.expand_dims(
                    tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(tf.reshape(self.genres_len, [-1]))), 1), -1)
                self.emb_genres = tf.reduce_sum(self.emb_genres * genres_mask, 1)

                self.emb_item = tf.concat([self.candidate_emb, self.emb_genres, self.avg_rating[:, tf.newaxis], self.year[:, tf.newaxis]], 1)
                self.candidate_bias = tf.matmul(self.emb_item, self.b_candidate_movie_id[:, tf.newaxis])
                self.emb_item = tf.layers.dense(self.emb_item, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_item = tf.layers.dense(self.emb_item, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_item = tf.layers.dense(self.emb_item, 32, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_item = tf.layers.dense(self.emb_item, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

            # elements wise dot of user and item embedding
            with tf.variable_scope("gmf"):
                self.gmf = tf.reduce_sum(self.emb_query * self.emb_item, 1, keep_dims=True)
                self.gmf = tf.add(self.gmf, self.b_global)
                self.gmf = tf.add(self.gmf, self.query_bias)
                self.gmf = tf.add(self.gmf, self.candidate_bias, name="infer")

                # one query for all items, for predict speed
                self.pred = tf.matmul(self.emb_query, tf.transpose(self.emb_item)) + \
                            tf.reshape(self.candidate_bias, (1, -1)) + \
                            self.query_bias + \
                            self.b_global
                self.pred = tf.nn.sigmoid(self.pred)

            with tf.variable_scope("loss"):
                self.alter_rating = tf.to_float(self.rating >= 4)[:, tf.newaxis]
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.alter_rating, logits=self.gmf))

            with tf.variable_scope("train"):
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                # self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
                pass

            self.saver = tf.train.Saver(tf.global_variables())
            self.graph = graph
            self.model_dir = model_dir

    def reset_model(self, model_dir):
        shutil.rmtree(path=model_dir, ignore_errors=True)
        os.makedirs(model_dir)

    def feed_dict(self, data, mode="train"):
        ret = {
            self.query_movie_ids: data["query_movie_ids"],
            self.query_movie_ids_len: data["query_movie_ids_len"],
            self.genres: data["genres"],
            self.genres_len: data["genres_len"],
            self.avg_rating: data["avg_rating"],
            self.year: data["year"],
            self.candidate_movie_id: data["candidate_movie_id"]
        }
        ret[self.is_train] = False
        if mode != "infer":
            ret[self.rating] = data["rating"]
            if mode == "train":
                ret[self.is_train] = True
            elif mode == "eval":
                pass
        return ret

    def fit(self, sess, trainGen, testGen, reset=False, n_epoch=50):
        sess.run(tf.global_variables_initializer())
        if reset:
            print("reset model: clean model dir: {} ...".format(self.model_dir))
            self.reset_model(self.model_dir)
        # try: 試著重上次儲存的model再次training
        self.ckpt(sess, self.model_dir)

        start = time.time()
        print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
        minLoss = 1e7
        for ep in range(1, n_epoch + 1):
            tr_loss, tr_total = 0, 0
            for i, data in enumerate(trainGen(), 1):
                loss, _ = sess.run([self.loss, self.train_op], feed_dict=self.feed_dict(data, mode="train"))
                batch_len = len(data["query_movie_ids"])
                tr_loss += loss * batch_len
                tr_total += batch_len
                print("\rtrain loss: {:.3f}".format(loss), end="")

            if testGen is not None:
                te_loss = self.epoch_loss(sess, testGen)

            tpl = "\r%02d\t%.3f\t\t%.3f\t\t%.3f secs"
            if minLoss > te_loss:
                tpl += ", saving ..."
                self.saver.save(sess, os.path.join(self.model_dir, 'model'), global_step=ep)
                minLoss = te_loss

            end = time.time()
            print(tpl % (ep, tr_loss / tr_total, te_loss, end - start))
            start = end
        return self

    def ckpt(self, sess, model_dir):
        """load latest saved model"""
        latestCkpt = tf.train.latest_checkpoint(model_dir)
        if latestCkpt:
            self.saver.restore(sess, latestCkpt)
        return latestCkpt

    def epoch_loss(self, sess, data_gen):
        tot_loss, tot_cnt = 0, 0
        for data in data_gen():
            loss_tensor = self.loss
            loss = sess.run(loss_tensor, feed_dict=self.feed_dict(data, mode="eval"))
            tot_loss += loss * len(data["query_movie_ids"])
            tot_cnt += len(data["query_movie_ids"])
        return tot_loss / tot_cnt

    def predict(self, sess, user_queries, items):
        self.ckpt(sess, self.model_dir)
        return sess.run(self.pred, feed_dict={
            self.is_train: False,
            self.query_movie_ids: user_queries["query_movie_ids"],
            self.query_movie_ids_len: user_queries["query_movie_ids_len"],

            self.genres: items["genres"],
            self.genres_len: items["genres_len"],
            self.avg_rating: items["avg_rating"],
            self.year: items["year"],
            self.candidate_movie_id: items["candidate_movie_id"]
        })

