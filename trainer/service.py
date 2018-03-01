import yaml, os, tensorflow as tf, json

from . import env
from . import reco_mf_dnn_est as est
from .utils import flex, utils

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery


seed = 88
class Service(object):
    logger = env.logger('Service')

    def __init__(self):
        self.storage = None

    def read_user_conf(self, conf_path):
        with flex.io(conf_path) as f:
            return yaml.load(f.as_reader().stream)

    def unser_parsed_conf(self, parsed_conf_path):
        with flex.io(parsed_conf_path).as_reader() as f:
            return flex.Schema.unserialize(f.stream)

    def find_raws(self, p):
        return flex.io(p.raw_dir).list()

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
            log_step_count_steps=300,
            tf_random_seed=seed,
            save_checkpoints_secs=10
            # save_checkpoints_steps=p.save_every_steps,
        )

        # try to build local export directory avoid error
        if p.is_local:
            import traceback
            try:
                os.makedirs(utils.join(p.job_dir, 'export', p.export_name))
            except:
                self.logger.error(traceback.format_exc())

        # TODO: hack, take off this property
        self.model = model
        model.fit(train_input, valid_input, run_config, reset=True)
        self.deploy()


    def deploy(self, pid, model_export_path):
        credentials = GoogleCredentials.get_application_default()
        ml = discovery.build('ml', 'vl', credentials=credentials)
        ml.projects().models().create(
            parent='projects/{}'.format(env.PROJECT_ID),
            body={
                'name': '{}_recommendation'.format(pid),
                'deploymentUri': model_export_path
            })


    def predict(self, p):
        self.logger.info('predict.params: {}'.format(p.values()))
        loader = flex.Loader(p.conf_path, p.parsed_conf_path)
        # transform data to model recognizable
        data = loader.trans_json(p.data)

        # ml-engine local predict
        tmpfile = 'tmp.{}.json'.format(utils.timestamp())
        # TODO:
        with flex.io(tmpfile).as_writer('w') as f:
            json.dump(data, f.stream)

        ml = discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}'.format(project, model)

        # command = ('gcloud ml-engine local predict'
        #            ' --model-dir D:/Python/notebook/recomm_prod/repo/foo/model_1518581106.1947258/export/export_foo/1518581138'
        #            ' --json-instances {}'.format(tmpfile))
        # self.logger.info(command)
        # utils.cmd(command)
