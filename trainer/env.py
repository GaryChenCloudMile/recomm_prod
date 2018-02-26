import os, yaml, logging, logging.config, codecs

# recommendation engine service bucket path
HQ_BUCKET = 'gs://recomm-job'

DATA = 'data'
MODEL = 'model'
LOG = 'log'

TRAIN_FNAME = 'data.tr'
VALID_FNAME = 'data.vl'

ERR_CDE = 'err_cde'
ERR_MSG = 'err_msg'

CREDENTIAL_NAME = 'GOOGLE_APPLICATION_CREDENTIALS'
# hack
print('os.environ[CREDENTIAL_NAME]:', os.environ.get(CREDENTIAL_NAME))
if CREDENTIAL_NAME not in os.environ:
    os.environ[CREDENTIAL_NAME] = '../auth.json'

PROJECT_PATH = 'D:/Python/notebook/recomm_prod'

class Logging(object):
    instance = None
    sd_handler = None

    @staticmethod
    def logger(name):
        if Logging.instance is None:
            with codecs.open(os.path.join(os.path.dirname(__file__), 'logging.yaml'), 'r', 'utf-8') as r:
                logging.config.dictConfig(yaml.load(r))
            Logging.instance = logging

        logger_ = Logging.instance.getLogger(name)
        # # stack driver client
        # if Logging.sd_handler is None and os.environ.get(CREDENTIAL_NAME) is not None:
        #     from google.cloud import logging as sd_logging
        #     Logging.sd_handler = sd_logging.Client().get_default_handler()
        #
        # # see if exists stack driver handler
        # if Logging.sd_handler is not None:
        #     logger_.addHandler(Logging.sd_handler)
        return logger_

# short path of Logging.logger
def logger(name):
    return Logging.logger(name)

class APIClient:
    storage_client = None

def bucket(bucket_name):
    if APIClient.storage_client is None:
        from google.cloud import storage
        APIClient.storage_client = storage.Client()
    return APIClient.storage_client.get_bucket(bucket_name)

