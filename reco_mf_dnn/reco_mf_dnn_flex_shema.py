import numpy as np, tensorflow as tf, os, time, pandas as pd, codecs, yaml, shutil

from datetime import datetime
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
    VOCABS = 'vocabs'
    VOCABS_PATH = 'vocabs_path'
    SEP = 'sep'
    AUX = 'aux'
    M_DTYPE_ARY = [CONT, CATG, DATETIME]
    TYPE = 'type'
    COL_STATE = 'col_state'

    COL_ATTR = [ID, M_DTYPE, DATE_FORMAT, DEFAULT, IS_MULTI,
                SEP, VOCABS, VOCABS_PATH, AUX, TYPE,
                COL_STATE]

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

    @property
    def raw_cols(self):
        return self.df_conf_.index.tolist()

    @property
    def cols(self):
        return list(self.col_states_.keys())

    @property
    def user_cols(self):
        return self.df_conf_.query("{} == '{}'".format(Schema.TYPE, Schema.USER)).id.tolist()

    @property
    def item_cols(self):
        return self.df_conf_.query("{} == '{}'".format(Schema.TYPE, Schema.ITEM)).id.tolist()

    @property
    def label(self):
        return self.df_conf_.query("{} == '{}'".format(Schema.TYPE, Schema.LABEL)).id.tolist()

    @property
    def tensor_type(self):
        dtypes = OrderedDict()
        df_conf = self.df_conf_.query("{}.notnull()".format(Schema.TYPE))
        for _, r in df_conf.iterrows():
            dtypes[r[Schema.ID]] = tf.int32 if r[Schema.M_DTYPE] == Schema.CATG else tf.float32
        return dtypes

    def init(self):
        return self.extract(self.read_conf()).check().fit()

    def read_conf(self):
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

        self.df_conf_ = self.df_conf_.set_index(Schema.ID, drop=False)
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

        # inference whether to add auxiliary col, like sequence, weighted columns
        return self.check_aux()

    def check_aux(self):
        # need_aux = self.df_conf_.query("{}.notnull() and {} == '{}' and {} == True" \
        #                         .format(Schema.TYPE, Schema.M_DTYPE, Schema.CATG, Schema.IS_MULTI))
        # # print('need_aux', need_aux)
        # for id, r in need_aux.iterrows():
        #     # COL_ATTR = [ID, M_DTYPE, DATE_FORMAT, DEFAULT, IS_MULTI,
        #     #             SEP, VOCABS, VOCABS_PATH, AUX, TYPE,
        #     #             COL_STATE]
        #     self.df_conf_ = self.df_conf_.append({Schema.ID: '{}_len'.format(id),
        #                                           Schema.M_DTYPE: Schema.CONT,
        #                                           Schema.IS_MULTI: False,
        #                                           Schema.AUX: True,
        #                                           Schema.TYPE: Schema.AUX}, ignore_index=True)
        # self.df_conf_ = self.df_conf_.where(self.df_conf_.notnull(), None).set_index('id', drop=False)
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
                                     names=self.raw_cols,
                                     chunksize=20000, dtype=dtype):

                chunk = chunk.where(pd.notnull(chunk), None)
                # loop all valid columns except label
                for _, r in df_conf.iterrows():
                    val, m_dtype, name, col_type = None, r[Schema.M_DTYPE], r[Schema.ID], r[Schema.TYPE]
                    is_aux = r[Schema.AUX]

                    if is_aux: continue

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
                            vocabs, vocabs_path = r[Schema.VOCABS], r[Schema.VOCABS_PATH]
                            if name not in col_states:
                                if is_multi:
                                    col_states[name] = utils.CatgMapper(name,
                                                                        is_multi=is_multi,
                                                                        sep=sep,
                                                                        vocabs=vocabs,
                                                                        vocabs_path=vocabs_path)\
                                                            .init_check()
                                else:
                                    col_states[name] = utils.CatgMapper(name,
                                                                        vocabs=vocabs,
                                                                        vocabs_path=vocabs_path)\
                                                            .init_check()
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
        valid_cond = self.df_conf_[Schema.TYPE].notnull() & (self.df_conf_[Schema.AUX] == False)
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

        # output type: catg+multi to str
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
        this.df_conf_ = this.df_conf_.set_index(Schema.ID, drop=False)
        # specific class for each type
        ser_maps = {Schema.CATG: utils.CatgMapper,
                    Schema.CONT: utils.NumericMapper,
                    Schema.DATETIME: utils.DatetimeMapper}

        this.col_states_ = OrderedDict()
        for _, r in this.df_conf_.query('{}.notnull() and {} == False'\
                                 .format(Schema.TYPE, Schema.AUX)).iterrows():
            this.col_states_[r[Schema.ID]] = \
                utils.BaseMapper.unserialize(ser_maps[r[Schema.M_DTYPE]], r[Schema.COL_STATE])
        return this


class Loader(object):
    def __init__(self, conf_path, parsed_conf_path, raw_paths:list=None):
        """

        :param conf_path:
        :param parsed_conf_path:
        :param raw_paths:
        """
        self.conf_path = conf_path
        self.parsed_conf_path = parsed_conf_path
        self.raw_paths = raw_paths
        self.schema = None

    def check_schema(self):
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
        return self

    def tansform(self, src_path, tgt_path, chunksize=20000, reset=False, valid_size=None):
        # reset: remove parsed config file and rebuild
        if reset:
            self.schema = None
            utils.rm_quiet(self.parsed_conf_path)

        self.check_schema()

        # TODO: alter print function to logging
        print('try to transform {} ... '.format(src_path))
        return self._transform(src_path, tgt_path, chunksize=chunksize, valid_size=valid_size)

    def _transform(self, src_path, tgt_path, chunksize=20000, valid_size=None):
        tr_tgt_path = '{}.tr'.format(tgt_path)
        vl_tgt_path = '{}.vl'.format(tgt_path)
        # delete first
        for file2del in (tgt_path, tr_tgt_path, vl_tgt_path): utils.rm_quiet(file2del)
        # if don't split, discard vl_tgt_path variable
        if not valid_size:
            tr_tgt_path = tgt_path

        df_conf = self.schema.df_conf_
        dtype = self.schema.raw_dtype(df_conf)
        columns = df_conf[Schema.ID].values
        col_states = self.schema.col_states_
        pos = 0

        trw, vlw, rand_seq = codecs.open(tr_tgt_path, 'a', 'utf-8'), None, None
        if valid_size:
            rand_seq = np.random.random(size=self.schema.count_)
            vlw = codecs.open(vl_tgt_path, 'a', 'utf-8')
        try:
            s = datetime.now()
            for step, chunk in enumerate(pd.read_csv(src_path, names=columns, chunksize=chunksize, dtype=dtype), 1):
                chunk = chunk.where(pd.notnull(chunk), None)[self.schema.cols]
                for colname, col in chunk.iteritems():
                    # multivalent categorical columns
                    if df_conf.loc[colname, Schema.M_DTYPE] == Schema.CATG and \
                       df_conf.loc[colname, Schema.IS_MULTI]:
                        val = pd.Series(col_states[colname].transform(col))
                        # because of persist to csv, transfer int array to string splitted by comma
                        chunk[colname] = val.map(lambda ary: ','.join(map(str, ary))).tolist()
                    # univalent columns
                    else:
                        chunk[colname] = list(col_states[colname].transform(col))

                kws = {'index': False, 'header': None}
                if not valid_size:
                    chunk.to_csv(trw, **kws)
                else:
                    end_pos = pos + len(chunk)
                    rand_batch = rand_seq[pos:end_pos]
                    # 1 - (valid_size * 100)% training data, (valid_size * 100)% testing data
                    tr_chunk, vl_chunk = chunk[rand_batch > valid_size], chunk[rand_batch <= valid_size]
                    tr_chunk.to_csv(trw, **kws)
                    vl_chunk.to_csv(vlw, **kws)
                    pos = end_pos

            # TODO: alter print function to logging
            print('[{}]: process take time {}'.format(src_path, datetime.now() - s))
        finally:
            _ = trw.close() if trw is not None else None
            _ = vlw.close() if vlw is not None else None
        return self

class ModelMfDNN(object):
    def __init__(self,
                 schema,
                 n_items,
                 n_genres,
                 model_dir,
                 dim=32):
        self.n_items = n_items
        self.n_genres = n_genres
        self.model_dir = model_dir
        self.dim = dim
        self.schema = schema

    def graph(self, features, labels, mode):
        with tf.variable_scope("inputs"):
            self.is_train = tf.placeholder(tf.bool, None)
            self.features, self.labels = features, labels
            for name, tensor in self.features.items():
                setattr(self, name, tensor)

        init_fn = tf.glorot_normal_initializer()
        emb_init_fn = tf.glorot_uniform_initializer()
        self.b_global = tf.Variable(emb_init_fn(shape=[]), name="b_global")

        with tf.variable_scope("embedding") as scope:
            self.w_query_movie_ids = tf.Variable(emb_init_fn(shape=[self.n_items, self.dim]), name="w_query_movie_ids")
            self.b_query_movie_ids = tf.Variable(emb_init_fn(shape=[self.dim]), name="b_query_movie_ids")
            self.w_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items, self.dim]), name="w_candidate_movie_id")
            self.b_candidate_movie_id = tf.Variable(init_fn(shape=[self.dim + 8 + 2]), name="b_candidate_movie_id")
            # self.b_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items]), name="b_candidate_movie_id")
            self.w_genres = tf.Variable(emb_init_fn(shape=[self.n_genres, 8]), name="w_genres")

        with tf.variable_scope("user_encoding") as scope:
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
        with tf.variable_scope("item_encoding") as scope:
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
        with tf.variable_scope("gmf") as scope:
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

        # hack
        # print('self.features.keys:', features.keys())
        # print('self.gmf:', self.gmf)
        # print('self.labels[:, tf.newaxis]:', self.labels[:, tf.newaxis])

        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            # export_outputs = {
            #     'predictions': tf.estimator.export.PredictOutput(self.pred)
            # }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=self.pred)

        with tf.variable_scope("loss") as scope:
            # self.alter_rating = tf.to_float(self.label >= 4)[:, tf.newaxis]
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.labels)[:, tf.newaxis], logits=self.gmf))

        self.train_op = None
        self.global_step = tf.train.get_or_create_global_step()
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("train"):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = tf.train.AdamOptimizer(0.005).minimize(self.loss, self.global_step)
                    # self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        # self.saver = tf.train.Saver(tf.global_variables())
        # self.graph = graph
        # self.model_dir = model_dir
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=None)

    def input_fn(self, filenames, n_batch=128, n_epoch=10, shuffle=True):
        cols = ['query_movie_ids', 'genres', 'avg_rating', 'year', 'candidate_movie_id', 'rating']
        defaults = [[''], [''], [], [], [0], [0]]
        multi_cols = ('query_movie_ids', 'genres')

        def _input_fn():
            def parse_csv(value):
                data = tf.decode_csv(value, record_defaults=defaults)
                features = OrderedDict(zip(cols, data))
                for col in multi_cols:
                    features[col] = tf.string_to_number(
                        tf.string_split([features[col]], ',').values, out_type=tf.int32)
                return features

            def add_seq_cols(feat):
                for m_col in multi_cols:
                    name = '{}_len'.format(m_col)
                    feat[name] = tf.size(feat[m_col])
                    cols.append(name)
                return feat

            dataset = tf.data.TextLineDataset(filenames)
            dataset = dataset.map(parse_csv, num_parallel_calls=4)
            dataset = dataset.map(add_seq_cols, num_parallel_calls=4)
            dataset = dataset.padded_batch(n_batch, OrderedDict(zip(cols, ([None], [None], [], [], [], [], [], [], []))))
            dataset = dataset.shuffle(n_batch * 1000, seed=seed).repeat(n_epoch)
            features = dataset.make_one_shot_iterator().get_next()
            return features, features.pop('rating')
        return _input_fn

    def serving_inputs(self):
        placeholders = OrderedDict()
        for name, tensor in self.features.items():
            placeholders[name] = tf.placeholder(shape=tensor.get_shape().as_list(), dtype=tensor.dtype)

        placeholders['labels'] = self.labels
        return tf.estimator.export.ServingInputReceiver(self.features, placeholders)


    def fit(self, train_input=None, valid_input=None, reset=True):
        if reset:
            print(self.model_dir)
            shutil.rmtree(self.model_dir)

        train_input = self.input_fn(['./movielens.tr'])
        valid_input = self.input_fn(['./movielens.vl'], n_epoch=1, shuffle=False)
        train_spec = tf.estimator.TrainSpec(train_input)
        exporter = tf.estimator.FinalExporter('movielens_export', self.serving_inputs)
        eval_spec = tf.estimator.EvalSpec(valid_input,
                                           # steps=(30287 // 128) + 1,
                                           exporters=[exporter],
                                           name='movielens_eval')

        config = tf.estimator.RunConfig(tf_random_seed=seed, save_checkpoints_steps=(69717 // 128) + 1)
        self.estimator_ = tf.estimator.Estimator(model_fn=self.graph, model_dir=self.model_dir, config=config)

        tf.estimator.train_and_evaluate(self.estimator_, train_spec, eval_spec)
        return self

    def predict(self, sess, user_queries, items):
        pass

