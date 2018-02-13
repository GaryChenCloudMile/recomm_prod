import yaml, codecs, os, env, tensorflow as tf

from reco_mf_dnn import reco_mf_dnn_est as est
from utils import flex

seed = 88

class Service(object):

    def __init__(self):
        self.logger = env.logger(__name__)

    def read_user_conf(self, conf_path):
        with codecs.open(conf_path, 'r', 'utf-8') as r:
            return yaml.load(r)

    def unser_parsed_conf(self, parsed_conf_path):
        with codecs.open(parsed_conf_path, 'r', 'utf-8') as r:
            return flex.Schema.unserialize(r)

    def find_raws(self, p):
        # TODO: change to GCS style
        return [os.path.join(root, f) for root, ds, fs in os.walk(p.raw_dir) for f in fs]

    def unserialize(self, fp):
        return flex.Schema.unserialize(fp)

    def gen_data(self, p):
        p.add_hparam('raw_paths', self.find_raws(p))
        assert len(p.raw_paths), 'must supply training data to processing! found nothing in {}' \
            .format(p.raw_dir)

        loader = flex.Loader(conf_path=p.conf_path,
                              parsed_conf_path=p.parsed_conf_path,
                              raw_paths=p.raw_paths)
        loader.transform(p.raw_paths, p.train_file, p.valid_file, reset=True, valid_size=.3)
        return loader.schema

    def train(self, p, schema):
        # hack
        from pprint import pprint
        from io import StringIO
        sio = StringIO()
        pprint(p.values(), sio)
        self.logger.info('hparam: {}'.format(sio.getvalue()))

        model = est.ModelMfDNN(hparam=p, schema=schema, n_items=9125, n_genres=20)
        train_input = model.input_fn([p.train_file], n_epoch=None, n_batch=p.n_batch)
        valid_input = model.input_fn([p.valid_file], n_epoch=None, n_batch=p.n_batch, shuffle=False)
        run_config = tf.estimator.RunConfig(
            log_step_count_steps=100,
            tf_random_seed=seed,
            # save_checkpoints_steps=p.save_every_steps,
            )

        model.fit(train_input, valid_input, run_config, reset=True)
        return model
