import argparse, yaml, codecs, env, os, shutil

from tensorflow.contrib.training.python.training.hparam import HParams

class Ctrl(object):

    PROJECT_ID = 'project_id'
    RAW_DIR = 'raw_dir'
    OVERRIDE = 'override'

    def process(self, params):
        with codecs.open(params.conf_path, 'r', 'utf-8') as r:
            conf = yaml.load(r)

        p = HParams(pid=conf[Ctrl.PROJECT_ID],
                    raw_dir=conf[Ctrl.RAW_DIR],
                    override=True if conf.get(Ctrl.OVERRIDE) else False)
        self.check_project(p)

        train_path = '/'.join([env.DATA_PATH, p.pid, 'train'])
        os.makedirs(train_path, exist_ok=True)

        p.train_files = ['/'.join([train_path, 'data.tr'])]
        p.valid_files = ['/'.join([train_path, 'data.vl'])]
        p.raw_paths = self.find_raws(p)


    def check_project(self, p):
        # TODO: change to GCS style
        model_path = '/'.join([env.MODEL_PATH, p.pid])
        data_path = '/'.join([env.DATA_PATH, p.pid])
        if not p.override and os.path.exists(model_path):
            print('project id [{}] exists, put [override: true] in config file for overriding!'
                  .format(p.pid))
            return self

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        p.job_dir = model_path
        return self

    def find_raws(self, p):
        # TODO: change to GCS style
        return ['{}/{}'.format(root, f) for root, ds, fs in os.walk(p.raw_dir) for f in fs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf-path',
        help='config path in GCS',
    )

    params = HParams(**parser.parse_args().__dict__)
    Ctrl().process(params)