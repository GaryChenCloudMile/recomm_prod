import tensorflow as tf, os, time, traceback
import trainer.env as env

from collections import OrderedDict

seed = 88

class ModelMfDNN(object):
    def __init__(self,
                 hparam=None,
                 schema=None,
                 n_items=None,
                 n_genres=None):
        self.n_items = n_items
        self.n_genres = n_genres
        self.schema = schema
        self.hparam = hparam
        self.model_dir = hparam.job_dir
        self.logger = env.logger('ModelMfDNN')

    def graphing(self, features, labels, mode):
        p = self.hparam
        with tf.variable_scope('inputs') as scope:
            self.features, self.labels = features, labels
            for name, tensor in self.features.items():
                setattr(self, name, tensor)

        with tf.variable_scope("init") as scope:
            init_fn = tf.glorot_normal_initializer()
            emb_init_fn = tf.glorot_uniform_initializer()
            self.b_global = tf.Variable(emb_init_fn(shape=[]), name="b_global")

            with tf.variable_scope("embedding") as scope:
                self.w_query_movie_ids = tf.Variable(emb_init_fn(shape=[self.n_items, p.dim]), name="w_query_movie_ids")
                self.b_query_movie_ids = tf.Variable(emb_init_fn(shape=[p.dim]), name="b_query_movie_ids")
                self.w_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items, p.dim]), name="w_candidate_movie_id")
                self.b_candidate_movie_id = tf.Variable(init_fn(shape=[p.dim + 8 + 2]), name="b_candidate_movie_id")
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
            # self.emb_query = tf.layers.dense(self.emb_query, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

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
            # self.emb_item = tf.layers.dense(self.emb_item, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

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

        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'outputs': tf.estimator.export.PredictOutput({
                    'emb_query': self.emb_query,
                    'emb_item': self.emb_item,
                    'pred': self.pred
                })
            }
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=self.pred,
                                              export_outputs=export_outputs)

        with tf.variable_scope("loss") as scope:
            # self.alter_rating = tf.to_float(self.label >= 4)[:, tf.newaxis]
            self.ans = tf.to_float(self.labels)[:, tf.newaxis]
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ans, logits=self.gmf))

        with tf.variable_scope("metrics") as scope:
            self.auc = tf.metrics.auc(tf.cast(self.labels, tf.bool),
                                      tf.reshape(tf.nn.sigmoid(self.gmf), [-1]))

        self.train_op = None
        self.global_step = tf.train.get_or_create_global_step()
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("train"):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = tf.train.AdamOptimizer().minimize(self.loss, self.global_step)
                    # self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        # self.merge = tf.summary.merge_all()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops={'auc': self.auc},
            evaluation_hooks=[])

    def input_fn(self, filenames, n_batch=128, n_epoch=None, shuffle=True):
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
            if shuffle:
                dataset = dataset.shuffle(n_batch * 10, seed=seed)
            dataset = dataset.repeat(n_epoch)
            dataset = dataset.padded_batch(n_batch, OrderedDict(zip(cols, ([None], [None], [], [], [], [], [], [], []))))
            features = dataset.make_one_shot_iterator().get_next()
            return features, features.pop('rating')
        return _input_fn

    def serving_inputs(self):
        placeholders = OrderedDict()
        for name, tensor in self.features.items():
            placeholders[name] = tf.placeholder(shape=tensor.get_shape().as_list(), dtype=tensor.dtype)

        placeholders['labels'] = self.labels
        return tf.estimator.export.ServingInputReceiver(placeholders, placeholders)


    def fit(self, train_input, valid_input, run_config, reset=True):
        if reset:
            # TODO:
            # print('clear checkpoint directory {}'.format(self.model_dir))
            # shutil.rmtree(self.model_dir)
            self.model_dir = '{}_{}'.format(self.model_dir, time.time())
            os.makedirs(self.model_dir)

        p = self.hparam
        # summary_hook = tf.train.SummarySaverHook(
        #     100, output_dir=self.model_dir, summary_op=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
        train_spec = tf.estimator.TrainSpec(train_input, max_steps=p.train_steps, hooks=None)
        # exporter = tf.estimator.LatestExporter(p.export_name, self.serving_inputs)
        exporter = BestScoreExporter(p.export_name, self.serving_inputs)
        eval_spec = tf.estimator.EvalSpec(valid_input,
                                          steps=p.eval_steps,
                                          exporters=[exporter],
                                          name=p.eval_name,
                                          # throttle_secs=26
                                         )
        # try to build local export directory avoid error
        # TODO:
        try:
            os.makedirs( os.path.join(self.model_dir, 'export', p.export_name) )
        except:
            print( traceback.format_exc() )

        self.estimator_ = tf.estimator.Estimator(model_fn=self.graphing, model_dir=self.model_dir, config=run_config)
        tf.estimator.train_and_evaluate(self.estimator_, train_spec, eval_spec)
        return self

    def predict(self, sess, user_queries, items):
        pass


class MyHook(tf.train.SessionRunHook):
    def __init__(self, tensor):
        self.tensor = tensor

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.tensor)

    def after_run(self, run_context, run_values):
        print(len(run_values.results))

class BestScoreExporter(tf.estimator.Exporter):
    def __init__(self,
                 name,
                 serving_input_receiver_fn,
                 assets_extra=None,
                 as_text=False):
        self._name = name
        self.serving_input_receiver_fn = serving_input_receiver_fn
        self.assets_extra = assets_extra
        self.as_text = as_text
        self.best = None
        self.logger = env.logger('BestScoreExporter')
        print('BestScoreExporter init')

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):

        curloss = eval_result['loss']
        if self.best is None or self.best >= curloss:
            self.best = curloss
            self.logger.info('nice eval loss: {}, export to pb'.format(curloss))
            estimator.export_savedmodel(
                export_path,
                self.serving_input_receiver_fn,
                assets_extra=self.assets_extra,
                as_text=self.as_text,
                checkpoint_path=checkpoint_path)
        else:
            self.logger.info('bad eval loss: {}'.format(curloss))