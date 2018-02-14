import argparse, env, os, service, traceback

from datetime import datetime
from tensorflow.contrib.training.python.training.hparam import HParams

class Ctrl(object):
    instance = None

    PROJECT_ID = 'project_id'
    RAW_DIR = 'raw_dir'
    OVERRIDE = 'override'
    GCS = 'gcs'
    PARSED_FNAME = 'parsed.yaml'

    def __init__(self):
        self.service = service.Service()
        self.logger = env.logger('Ctrl')

    def pre_action(self, params, is_train=False):
        conf = self.service.read_user_conf(params.conf_path)
        p = HParams(conf_path=params.conf_path,
                    pid=conf[Ctrl.PROJECT_ID],
                    raw_dir=conf[Ctrl.RAW_DIR],
                    override=True if conf.get(Ctrl.OVERRIDE) else False)
        self.check_project(p, conf, is_train=is_train)

        os.makedirs(p.job_dir, exist_ok=True)
        os.makedirs(p.data_dir, exist_ok=True)
        p.add_hparam('train_file', os.path.join(p.repo, env.DATA, env.TRAIN_FNAME))
        p.add_hparam('valid_file', os.path.join(p.repo, env.DATA, env.VALID_FNAME))
        return p

    def check_project(self, p, conf=None, is_train=False):
        # TODO: change to GCS style
        # central repo
        p.add_hparam('repo', os.path.join(env.GCS, p.pid))

        # individual gcs from clients
        # p.add_hparam('repo', os.path.join(conf[Ctrl.GCS], p.pid))
        # p.add_hparam('log_dir', os.path.join(p.repo, env.LOG))
        # self.logger = env.logger(p.pid)
        if is_train and not p.override and os.path.exists(p.repo):
            raise Exception('project id [{}] exists, put [override: true] in config file for overriding!'
                            .format(p.pid))

        p.add_hparam('job_dir', os.path.join(p.repo, env.MODEL))
        p.add_hparam('data_dir', os.path.join(p.repo, env.DATA))
        p.add_hparam('parsed_conf_path', os.path.join(p.data_dir, Ctrl.PARSED_FNAME))
        return self

    def gen_data(self, params):
        ret = {}
        s = datetime.now()
        try:
            p = self.pre_action(params)
            self.service.gen_data(p)

            ret[env.ERR_CDE] = 00
        except Exception as e:
            ret[env.ERR_CDE] = 99
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
            # raise Exception(e)
        finally:
            # TODO:
            self.logger.info('{}: gen_data take time {}'.format(p.pid, datetime.now() - s))
        return ret

    def train(self, params):
        """do model ml-engine traning

        1. check if there's some data missing, if so, try to re generate training data and config file
        2. training recommendation model
        :param params: tensorflow HParams object storing user request data
        :return: json message
        """
        ret = {}
        try:
            p = self.pre_action(params, is_train=True)
            schema = None
            try:
                # TODO:
                assert os.path.exists(p.parsed_conf_path), \
                    'parsed config [{}] not found'.format(p.parsed_conf_path)

                for trf in (p.train_file, p.valid_file):
                    assert os.path.exists(trf), "training file [{}] not found".format(trf)
            except Exception as e:
                self.logger.warn(e)
                # try to gen training data
                # TODO:
                self.logger.info('{}: try to generate training data...'.format(p.pid))
                schema = self.service.gen_data(p)

            if schema is None:
                # TODO:
                self.logger.info('{}: try to unserialize {}'.format(p.pid, p.parsed_conf_path))
                schema = self.service.unser_parsed_conf(p.parsed_conf_path)

            p.add_hparam('export_name', 'export_{}'.format(p.pid))
            p.add_hparam('eval_name', '{}'.format(p.pid))
            p.add_hparam('n_batch', 128)

            ## runtime calculating attrs
            # training about 10 epochs
            tr_steps = self.count_steps(schema.tr_count_, p.n_batch)
            vl_steps = self.count_steps(schema.vl_count_, p.n_batch)
            p.add_hparam('train_steps', tr_steps * 3)
            p.add_hparam('eval_steps', vl_steps)
            p.add_hparam('dim', 16)
            # save once per epoch
            p.add_hparam('save_every_steps', None)
            model = self.service.train(p, schema)

            ret[env.ERR_CDE] = 00
            return model
        except Exception as e:
            ret[env.ERR_CDE] = 99
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            pass

        return ret

    def predict(self, params):
        conf = self.service.read_user_conf(params.conf_path)
        p = HParams(pid=conf[Ctrl.PROJECT_ID])
        self.check_project(p)
        p.add_hparam('parsed_conf_path', os.path.join(p.data_dir, Ctrl.PARSED_FNAME))


    def count_steps(self, n_total, n_batch):
        return n_total // n_batch + (1 if n_total % n_batch else 0)



# mock singleton
Ctrl.instance = Ctrl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf-path',
        help='config path in user\'s GCS',
    )

    Ctrl.instance.train(parser.parse_args())