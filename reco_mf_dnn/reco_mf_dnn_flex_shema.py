import numpy as np, tensorflow as tf, os, time, pandas as pd, codecs, yaml, shutil

from io import StringIO
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
    # N_UNIQUE = 'n_unique'
    VOCAB = 'vocab'
    VOCAB_PATH = 'vocab_path'
    SEP = 'sep'
    AUX = 'aux'
    M_DTYPE_ARY = [CONT, CATG, DATETIME]
    TYPE = 'type'
    COL_STATE = 'col_state'

    COL_ATTR = [ID, M_DTYPE, DATE_FORMAT, DEFAULT, IS_MULTI, SEP, AUX, TYPE, COL_STATE]
    # COL_TYPE = [np.str, np.str, np.str, 'object', np.bool, np.str, np.bool, np.str, np.str]
    # DEFAULT_VAL  = ['', '', '', '', False, '', False, '', '']

    def __init__(self, conf_path, parsed_conf_path, raw_paths:list):
        """Schema configs

        :param conf_path: path for config file columns specs configurations
        :param parsed_conf_path: path for parsed config file configurations
        :param raw_paths: multiple raw training csv files
        """
        self.conf_path = conf_path
        self.parsed_conf_path = parsed_conf_path
        # TODO: wait for fetch GCS training data to parse
        self.raw_paths = raw_paths

        self.count_ = 0
        self.conf_ = None
        self.df_conf_ = None
        self.col_states_ = None

    def init(self):
        return self.extract(self.parse_conf()).check().fit()

    def parse_conf(self):
        # TODO: wait for fetch gcs config file and read ...
        import codecs
        with codecs.open(self.conf_path, 'r', encoding='utf-8') as r:
            return r.read()

    def extract(self, conf):
        self.conf_ = yaml.load(conf)
        for k in (Schema.USER, Schema.ITEM, Schema.LABEL, Schema.COLUMNS):
            assert k in self.conf_, 'config requires {} attrs, actual {}'\
                .format([Schema.USER, Schema.ITEM, Schema.LABEL, Schema.COLUMNS], list(self.conf_.keys()))

        cols = []
        for r in self.conf_[Schema.COLUMNS]:
            cols.append(r)

        self.df_conf_ = (pd.DataFrame(columns=Schema.COL_ATTR, data=cols)
                           .reset_index(drop=True))
        self.df_conf_ = self.df_conf_.where(self.df_conf_.notnull(), None)
        # self.df_conf_.set_index('id', drop=False)

        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['item']), 'type'] = 'item'
        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['user']), 'type'] = 'user'
        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['label']), 'type'] = 'label'

        # is_multi, aux columns default False
        for col in (Schema.IS_MULTI, Schema.AUX):
            self.df_conf_[col].where(self.df_conf_[col].notnull(), False, inplace=True)
        return self

    def check(self):
        """check user input columns configs

        :return: self
        """

        # check if basic attr exists
        df_conf = self.df_conf_.query("type.notnull()")
        base = df_conf.query("{}.isnull() or {}.isnull()".format(Schema.ID, Schema.M_DTYPE))
        assert len(base) == 0, 'require {} attrs, check following settings:\n{}'\
            .format([Schema.ID, Schema.M_DTYPE], base)

        # check if user, item, label columns in self.conf_['columns']
        for k in (Schema.USER, Schema.ITEM, Schema.LABEL):
            unknowns = set(self.conf_[k]) - set(df_conf[Schema.ID])
            assert len(unknowns) == 0, '[{}]: {} not found in columns id settings'.format(k, list(unknowns))

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
        dt_no_format = df_conf.query("{} == '{}' and {}.isnull()"\
                              .format(Schema.M_DTYPE, Schema.DATETIME, Schema.DATE_FORMAT))
        assert len(dt_no_format) == 0, '{} column expect {} attr, check following:\n{}' \
            .format(Schema.DATETIME, Schema.DATE_FORMAT, dt_no_format)
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

        multi_no_sep = catg.query("{} == True and {}.isnull()".format(Schema.IS_MULTI, Schema.SEP))
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

        df_conf = self.df_conf_.query("{}.notnull()".format(Schema.TYPE))
        dtype = self.raw_dtype(df_conf)
        col_states = OrderedDict()
        # './merged_movielens.csv'
        for fpath in self.raw_paths:
            if not os.path.exists(fpath):
                # TODO: wait for logging object
                print("{} doesn't exists".format(fpath))
                continue

            for chunk in pd.read_csv(fpath,
                                     names=df_conf[Schema.ID].values,
                                     chunksize=10000, dtype=dtype):

                chunk = chunk.where(pd.notnull(chunk), None)
                # loop all valid columns except label
                for _, r in df_conf.iterrows():
                    val, m_dtype, name, col_type = None, r[Schema.M_DTYPE], r[Schema.ID], r[Schema.TYPE]

                    if col_type == Schema.LABEL:
                        if name not in col_states:
                            col_states[name] = utils.CatgMapper(name, allow_null=False).init_check()
                        # null value is not allowed in label column
                        assert not chunk[name].hasnans, 'null value detected in label column! filename {}' \
                            .format(fpath)
                    else:
                        # categorical column
                        if m_dtype == Schema.CATG:
                            is_multi, sep = r[Schema.IS_MULTI], r[Schema.SEP]
                            if name not in col_states:
                                if is_multi:
                                    col_states[name] = utils.CatgMapper(name, is_multi=is_multi, sep=sep)\
                                                            .init_check()
                                else:
                                    col_states[name] = utils.CatgMapper(name).init_check()
                        # numeric column
                        elif m_dtype == Schema.CONT:
                            if name not in col_states:
                                col_states[name] = utils.NumericMapper(name, default=r[Schema.DEFAULT])\
                                                        .init_check()
                        # datetime column: transform to numeric
                        elif m_dtype == Schema.DATETIME:
                            dt_fmt = r[Schema.DATE_FORMAT]
                            if name not in col_states:
                                col_states[name] = utils.DatetimeMapper(name, dt_fmt, default=r[Schema.DEFAULT])\
                                                        .init_check()
                    if m_dtype == Schema.CATG:
                        # if freeze_ == True, that means user provided vocabs informations,
                        # no need to fit anymore
                        if not col_states[name].freeze_:
                            col_states[name].partial_fit(chunk[name].values)
                    else:
                        col_states[name].partial_fit(chunk[name].values)

                # count data size
                self.count_ += len(chunk)

        self.col_states_ = col_states
        valid_cond = self.df_conf_[Schema.TYPE].notnull()
        # serialize parsed column states
        def ser(id):
            sio = StringIO()
            col_states[id].serialize(sio)
            return sio.getvalue()

        self.df_conf_.loc[valid_cond, 'col_state'] = \
            self.df_conf_.loc[valid_cond, Schema.ID].map(ser)

        # serialize to specific path
        with codecs.open(self.parsed_conf_path, 'w', 'utf-8') as w:
            self.serialize(w)
        return self

    def serialize(self, fp):
        """

        :param fp:
        :return:
        """
        return yaml.dump({
            'conf_path': self.conf_path,
            'parsed_conf_path': self.parsed_conf_path,
            'raw_paths': self.raw_paths,
            'count_': self.count_,
            'conf_': self.conf_,
            'df_conf_': self.df_conf_.to_dict(orient='records'),
        }, fp)

    @staticmethod
    def unserialize(fp):
        """Create Schema instance from config resource

        :param fp:
        :return:
        """
        info = yaml.load(fp)
        this = Schema(info['conf_path'], info['parsed_conf_path'], info['raw_paths'])
        for k, attr in info.items():
            setattr(this, k, attr)

        this.df_conf_ = pd.DataFrame(this.df_conf_, columns=Schema.COL_ATTR)
        # specific class for each type
        ser_maps = {Schema.CATG: utils.CatgMapper,
                    Schema.CONT: utils.NumericMapper,
                    Schema.DATETIME: utils.DatetimeMapper}

        this.col_states_ = OrderedDict()
        for _, r in this.df_conf_.query("type.notnull()").iterrows():
            this.col_states_[r[Schema.ID]] = \
                utils.BaseMapper.unserialize(ser_maps[r[Schema.M_DTYPE]], r[Schema.COL_STATE])
        return this

# class Conf(
#     collections.namedtuple("Conf",
#                            ("initializer", "source", "target_input",
#                             "target_output", "source_sequence_length",
#                             "target_sequence_length"))):
#     pass


class Loader(object):
    def __init__(self, conf_path, parsed_conf_path, raw_paths:list=None):
        self.conf_path = conf_path
        self.parsed_conf_path = parsed_conf_path
        self.raw_paths = raw_paths
        self.schema = None

    def load(self, data_path, reset=False):
        # reset: remove parsed config file and rebuild
        if reset:
            self.schema = None
            for file2del in (self.parsed_conf_path, data_path):
                if os.path.exists(file2del):
                    os.remove(file2del)

        # init schema
        if self.schema is None:
            # 1. try unserialize
            if os.path.isfile(self.parsed_conf_path):
                # TODO: alter print function to logging
                print('try to unserialize from {}'.format(self.parsed_conf_path))
                with codecs.open(self.parsed_conf_path, 'r', 'utf-8') as r:
                    self.schema = Schema.unserialize(r)
            # 2. if parsed_conf_path not exists, try re-parse raw config file (conf_path supplied by user)
            else:
                # TODO: alter print function to logging
                print('try to parse {} (user supplied) ...'.format(self.conf_path))
                self.schema = Schema(self.conf_path, self.parsed_conf_path, self.raw_paths).init()

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
            for fpath in self.raw_paths:
                # TODO: alter print function to logging
                print('process {} ...'.format(fpath))
                # hack 100 records test
                for chunk in pd.read_csv(fpath, names=columns, chunksize=100, dtype=dtype):
                    chunk = chunk.where(pd.notnull(chunk), None)
                    # hack
                    # chunk.to_csv('compare.csv', index=False)
                    chunk = chunk[list(col_states.keys())]
                    for colname, col in chunk.iteritems():
                        # multivalent categorical columns
                        if df_conf.loc[colname, Schema.M_DTYPE] == Schema.CATG and \
                           df_conf.loc[colname, Schema.IS_MULTI]:
                            val = pd.Series(col_states[colname].transform(col))
                            # because of persisting to csv, transfer int array to string splitted by comma
                            chunk[colname] = val.map(lambda ary: ','.join(map(str, ary))).tolist()
                        # univalent columns
                        else:
                            chunk[colname] = list(col_states[colname].transform(col))
                    chunk.to_csv(w, index=False)
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

