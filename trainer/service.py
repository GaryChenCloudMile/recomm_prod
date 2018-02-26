import yaml, codecs, os, tensorflow as tf, json, re

from . import env
from . import reco_mf_dnn_est as est
from .utils import flex, utils
from datetime import datetime
from io import BytesIO

seed = 88
class Service(object):
    logger = env.logger('Service')

    def __init__(self):
        self.storage = None

    def read_user_conf(self, conf_path):
        bucket, rest = utils.parse_gsc_uri(conf_path)
        return yaml.load(env.bucket(bucket).get_blob(rest).download_as_string())

    def unser_parsed_conf(self, parsed_conf_path):
        self.logger.info('try to unserialize from {}'.format(parsed_conf_path))
        parsed_conf = utils.gcs_blob(parsed_conf_path)
        return flex.Schema.unserialize(BytesIO(parsed_conf.download_as_string()))

    def find_raws(self, p):
        client_bucket, prefix = utils.parse_gsc_uri(p.raw_dir)
        return ['gs://{}/{}'.format(client_bucket, e.name)
                for e in env.bucket(client_bucket).list_blobs(prefix=prefix)]

    def gen_data(self, p):
        p.add_hparam('raw_paths', self.find_raws(p))
        assert len(p.raw_paths), 'must supply training data to processing! found nothing in {}' \
            .format(p.raw_dir)

        loader = flex.Loader(conf_path=p.conf_path,
                              parsed_conf_path=p.parsed_conf_path,
                              raw_paths=p.raw_paths)

        loader.transform(p, reset=False, valid_size=.3)
        return loader.schema

    def train(self, p, schema):
        # TODO: hack
        from pprint import pprint
        from io import StringIO
        sio = StringIO()
        pprint(p.values(), sio)
        self.logger.info('hparam: {}'.format(sio.getvalue()))

        model = est.ModelMfDNN(hparam=p, schema=schema, n_items=9125, n_genres=20)
        train_input = model.input_fn([p.train_file], n_epoch=1, n_batch=p.n_batch)
        valid_input = model.input_fn([p.valid_file], n_epoch=1, n_batch=p.n_batch, shuffle=False)
        run_config = tf.estimator.RunConfig(
            log_step_count_steps=100,
            tf_random_seed=seed,
            # save_checkpoints_steps=p.save_every_steps,
            )

        model.fit(train_input, valid_input, run_config, reset=True)
        return model

    def predict(self, p):
        self.logger.info('predict.params: {}'.format(p.values()))
        loader = flex.Loader(p.conf_path, p.parsed_conf_path)
        # transform data to model recognizable
        data = loader.trans_json(p.data)
        # hack
        # for k, v, in data.items():
        #     if k in ('query_movie_ids', 'genres '):
        #         print(k, type(v[0][0]))

        # ml-engine local predict
        tmpfile = 'tmp.{}.json'.format(datetime.now().strftime('%Y%m%d%H%M%S%f'))
        # TODO:
        with codecs.open(tmpfile, 'w', 'utf-8') as w:
            json.dump(data, w)

        command = ('gcloud ml-engine local predict'
                   ' --model-dir D:/Python/notebook/recomm_prod/repo/foo/model_1518581106.1947258/export/export_foo/1518581138'
                   ' --json-instances {}'.format(tmpfile))
        self.logger.info(command)
        utils.cmd(command)
