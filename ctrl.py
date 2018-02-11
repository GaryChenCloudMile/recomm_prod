import argparse, yaml, codecs, env, os, shutil

from tensorflow.contrib.training.python.training.hparam import HParams

class Ctrl(object):

    PROJECT_ID = 'project_id'
    RAW_DIR = 'raw_dir'
    OVERRIDE = 'override'
    GCS = 'gcs'

    def process(self, params):
        with codecs.open(params.conf_path, 'r', 'utf-8') as r:
            conf = yaml.load(r)

        p = HParams(pid=conf[Ctrl.PROJECT_ID],
                    raw_dir=conf[Ctrl.RAW_DIR],
                    override=True if conf.get(Ctrl.OVERRIDE) else False)
        self.check_project(p, conf)

        p.add_hparam('train_files', os.path.join(p.repo, env.DATA, env.TRAIN_FNAME))
        p.add_hparam('valid_files', os.path.join(p.repo, env.DATA, env.VALID_FNAME))
        p.add_hparam('raw_paths', self.find_raws(p))
        p.add_hparam('export_name', 'export')
        p.add_hparam('eval_name', 'eval')
        p.add_hparam('batch_size', 128)
        # runtime calculating attrs
        p.add_hparam('train_steps', 1000)
        p.add_hparam('eval_steps', 100)
        p.add_hparam('dim', 16)
        p.add_hparam('save_every_steps', 500)



    def check_project(self, p, conf):
        # TODO: change to GCS style
        # central repo
        # p.add_hparam('repo', os.path.join(env.GCS))

        # individual gcs from clients
        p.add_hparam('repo', os.path.join(conf[Ctrl.GCS], p.pid))
        # p.repo = os.path.join(conf[Ctrl.GCS], p.pid)
        if not p.override and os.path.exists(p.repo,):
            print('project id [{}] exists, put [override: true] in config file for overriding!'
                  .format(p.pid))
            return self

        p.add_hparam('job_dir', os.path.join(p.repo, env.MODEL))
        p.add_hparam('data_dir', os.path.join(p.repo, env.DATA))

        os.makedirs(p.job_dir, exist_ok=True)
        os.makedirs(p.data_dir, exist_ok=True)
        return self

    def find_raws(self, p):
        # TODO: change to GCS style
        return [os.path.join(root, f) for root, ds, fs in os.walk(p.raw_dir) for f in fs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf-path',
        help='config path in user\'s GCS',
    )

    params = HParams(**parser.parse_args().__dict__)
    Ctrl().process(params)