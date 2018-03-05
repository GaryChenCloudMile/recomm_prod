import yaml, os, tensorflow as tf, json

from . import env
from . import reco_mf_dnn_est as est
from .utils import flex, utils

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors


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
            # save_checkpoints_secs=None,
            # save_checkpoints_steps=p.save_every_steps,
        )

        # try to build local export directory avoid error
        if p.is_local:
            flex.io(utils.join(p.job_dir, 'export', p.export_name)).mkdirs()
            # os.makedirs(utils.join(p.job_dir, 'export', p.export_name))

        model.fit(train_input, valid_input, run_config, reset=True)

        # export deploy info
        deploy_info = {}
        deploy_info['job_id'] = p.job_id
        deploy_info['model_id'] = p.model_id
        deploy_info['export_path'] = model.exporter.export_result.decode()
        with flex.io(p.deploy_path).as_writer('w') as f:
            yaml.dump(deploy_info, f.stream)

        return model

    def deploy(self, p, export_path):
        credentials = GoogleCredentials.get_application_default()
        ml = discovery.build('ml', 'v1', credentials=credentials)
        model_name = '{}_{}'.format(p.pid, p.model_id).replace('-', '_')
        # res = ml.projects().models().versions().list(
        #     parent='projects/{}/models/{}'.format(env.PROJECT_ID, model_name)
        # ).execute()

        # ml.projects().models().create(
        #     parent='projects/{}'.format(env.PROJECT_ID),
        #     body={'name': model_name}
        # ).execute()

        ver_name = 'projects/{}/models/{}/versions/v1'.format(env.PROJECT_ID, model_name)
        ml.projects().models().versions().delete(name=ver_name)
        ml.projects().models().versions().create(
            parent=ver_name,
            body={
                'name': 'v1',
                'description': '[{}] recommendation model'.format(p.pid),
                # 'isDefault': True,
                'deploymentUri': export_path.decode() if isinstance(export_path, bytes) else export_path,
                'runtimeVersion': p.runtime_version,
                'state': 'UPDATING'
            }
        ).execute()

        # try:
        #     res = ml.projects().models().versions().get(
        #         name='projects/{}/models/{}/versions/{}'.format(env.PROJECT_ID, model_name, 'v12')
        #     ).execute()
        # except errors.HttpError as e:
        #     print('xxxxxxxx', vars(e))
        return res


    def predict(self, p):
        self.logger.info('predict.params: {}'.format(p.values()))
        loader = flex.Loader(p.conf_path, p.parsed_conf_path)
        # transform data to model recognizable
        data = loader.trans_json(p.data)

        # ml-engine local predict
        tmpfile = 'tmp.{}.json'.format(utils.timestamp())
        with flex.io(tmpfile).as_writer('w') as f:
            json.dump(data, f.stream)

        ml = discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}'.format(env.PROJECT_ID, p.jobid)

        # command = ('gcloud ml-engine local predict'
        #            ' --model-dir D:/Python/notebook/recomm_prod/repo/foo/model_1518581106.1947258/export/export_foo/1518581138'
        #            ' --json-instances {}'.format(tmpfile))
        # self.logger.info(command)
        # utils.cmd(command)
