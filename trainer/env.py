import os, yaml, logging, logging.config, codecs

# recommendation engine service GCS path
# GCS = 'D:/Python/notebook/recomm_prod/repo'
GCS = 'gs://recomm-job'

DATA = 'data'
MODEL = 'model'
LOG = 'log'

TRAIN_FNAME = 'data.tr'
VALID_FNAME = 'data.vl'

ERR_CDE = 'err_cde'
ERR_MSG = 'err_msg'


class Logging(object):
    instance = None

    @staticmethod
    def logger(name):
        if Logging.instance is None:
            log_conf = 'gs://recomm-job/log/logging.yaml'

            with codecs.open(os.path.join(os.path.dirname(__file__), 'logging.yaml'), 'r', 'utf-8') as r:
                logging.config.dictConfig(yaml.load(r))
            Logging.instance = logging
        return Logging.instance.getLogger(name)

# short path of Logging.logger
def logger(name):
    return Logging.logger(name)