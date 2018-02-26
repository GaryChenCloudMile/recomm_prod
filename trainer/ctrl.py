import argparse, os, time, yaml

from . import env, service
from .utils import utils
from datetime import datetime
from tensorflow.contrib.training.python.training.hparam import HParams

class Ctrl(object):
    instance = None

    PROJECT_ID = 'project_id'
    RAW_DIR = 'raw_dir'
    OVERRIDE = 'override'
    GCS = 'gcs'
    BUCKET = 'bucket'
    PARSED_FNAME = 'parsed.yaml'

    logger = env.logger('Ctrl')

    def __init__(self):
        self.service = service.Service()

    def check_project(self, p):
        # central repo
        p.add_hparam('repo', utils.join(env.HQ_BUCKET, p.pid))

        # individual gcs from clients
        # p.add_hparam('repo', utils.join(conf[Ctrl.GCS], p.pid))
        # p.add_hparam('log_dir', utils.join(p.repo, env.LOG))
        # self.logger = env.logger(p.pid)
        # if is_train and not p.override and os.path.exists(p.repo):
        #     raise Exception('project id [{}] exists, put [override: true] in config file for overriding!'
        #                     .format(p.pid))

        p.add_hparam('job_dir', utils.join(p.repo, env.MODEL))
        p.add_hparam('data_dir', utils.join(p.repo, env.DATA))
        p.add_hparam('parsed_conf_path', utils.join(p.data_dir, self.PARSED_FNAME))
        return self

    def pre_action(self, params):
        conf = self.service.read_user_conf(params.conf_path)
        p = HParams(conf_path=params.conf_path,
                    pid=conf[self.PROJECT_ID],
                    raw_dir=conf[self.RAW_DIR],
                    override=True if conf.get(self.OVERRIDE) else False)
        self.check_project(p)

        p.add_hparam('train_file', utils.join(p.repo, env.DATA, env.TRAIN_FNAME))
        p.add_hparam('valid_file', utils.join(p.repo, env.DATA, env.VALID_FNAME))
        return p

    def gen_data(self, params):
        ret = {}
        s = datetime.now()
        p = None
        try:
            p = self.pre_action(params)
            self.service.gen_data(p)

            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            self.logger.info('{}: gen_data take time {}'.format(p.pid, datetime.now() - s))
        return ret

    def train_submit(self, params):
        # ctx = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project = 'training-recommendation-engine'
        ret = {}
        s = datetime.now()
        p = None
        try:
            p = self.pre_action(params)
            jobid = '{}_{}'.format(p.pid, utils.timestamp()).replace('-', '_')
            commands = """
                cd {} && \
                gcloud ml-engine jobs submit training {} \
                    --job-dir {} \
                    --runtime-version 1.4 \
                    --module-name trainer.ctrl \
                    --package-path trainer \
                    --region asia-east1 \
                    --config config.yaml \
                    -- \
                    --method train \
                    --conf-path {}
            """.strip().format(env.PROJECT_PATH, jobid, p.job_dir, p.conf_path)
            # authpath = utils.join(ctx, 'auth.json')
            # svc = discovery.build('ml', 'v1', credentials=GoogleCredentials.from_stream(authpath))
            # resp = svc.projects().jobs()\
            #           .create(parent='projects/{}'.format(project),
            #                   body={
            #                       'jobId': 'recomm_movielens_16',
            #                       'trainingInput': {
            #                           'Module': 'trainer.ctrl',
            #                           'region': 'asia-east1',
            #                           'jobDir': 'gs://recomm-job/foo/model',
            #                           'packageUris': 'recomm-job/foo/model/packages/{}/package-0.0.0.tar.gz'.format(utils.timestamp()),
            #                           'runtimeVersion': '1.4',
            #                           'pythonVersion': '3.5'
            #                       }
            #                   })\
            #           .execute()
            ret['response'] = utils.cmd(commands)
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            self.logger.info('{}: gen_data take time {}'.format(p.pid, datetime.now() - s))
        return ret


    def train(self, params):
        """do model ml-engine traning

        :param params: tensorflow HParams object storing user request data
        :return: json message
        """
        ret = {}
        try:
            p = self.pre_action(params)
            schema = None
            try:
                parsed_conf = utils.gcs_blob(p.parsed_conf_path)
                assert parsed_conf.exists(), \
                    'parsed config [{}] not found'.format(p.parsed_conf_path)

                for trf in (p.train_file, p.valid_file):
                    blob = utils.gcs_blob(trf)
                    assert blob.exists(), "training file [{}] not found".format(trf)
            except Exception as e:
                raise e
                # try to gen training data
                # self.logger.info('{}: try to generate training data...'.format(p.pid))
                # schema = self.service.gen_data(p)

            if schema is None:
                self.logger.info('{}: try to unserialize {}'.format(p.pid, p.parsed_conf_path))
                schema = self.service.unser_parsed_conf(p.parsed_conf_path)

            p.add_hparam('export_name', 'export_{}'.format(p.pid))
            p.add_hparam('eval_name', '{}'.format(p.pid))
            p.add_hparam('n_batch', 128)

            if 'train_steps' not in p:
                # training about 10 epochs
                tr_steps = self.count_steps(schema.tr_count_, p.n_batch)
                p.add_hparam('train_steps', tr_steps * 10)

            if 'eval_steps' not in p:
                # training about 10 epochs
                vl_steps = self.count_steps(schema.vl_count_, p.n_batch)
                p.add_hparam('eval_steps', vl_steps)

            p.add_hparam('dim', 16)
            # save once per epoch, cancel this in case of saving bad model when encounter overfitting
            p.add_hparam('save_every_steps', None)
            model = self.service.train(p, schema)

            ret[env.ERR_CDE] = '00'
            return model
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            pass

        return ret

    def predict(self, params):
        ret = {}
        conf = self.service.read_user_conf(params.conf_path)
        p = HParams(pid=conf[Ctrl.PROJECT_ID], conf_path=params.conf_path, data=params.data)
        s = datetime.now()
        try:
            self.check_project(p)
            # TODO: GCS check if parsed conf path exists
            assert os.path.exists(p.parsed_conf_path), "can't find schema cause {} not exists" \
                .format(p.parsed_conf_path)

            ret['response'] = self.service.predict(p)
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
            # raise Exception(e)
        finally:
            # TODO:
            self.logger.info('{}: predict take time {}'.format(p.pid, datetime.now() - s))
        return ret

    def count_steps(self, n_total, n_batch):
        return n_total // n_batch + (1 if n_total % n_batch else 0)

    def load_schema(self, params):
        from .utils import flex
        p = self.pre_action(params)
        loader = flex.Loader(conf_path=p.conf_path,
                             parsed_conf_path=p.parsed_conf_path)
        loader.check_schema()
        return loader

    def test(self, params):
        pass


    def cmd(self, params):
        utils.cmd('gcloud ml-engine local predict'
                  ' --model_dir D:/Python/notebook/recomm_prod/repo/foo/model_1518581106.1947258/export/export_foo/1518581138'
                  ' --json-instance ')

    def cmd2(self, params):
        for i in range(1, 4):
            time.sleep(1)
            self.logger.info(i)



# mock singleton
Ctrl.instance = Ctrl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        help='execution method',
    )
    parser.add_argument(
        '--conf-path',
        help='config path in user\'s GCS',
    )
    parser.add_argument(
        '--job-dir',
        help='where to put checkpoints',
    )
    args = parser.parse_args()

    execution = getattr(Ctrl.instance, args.method)
    execution(args)