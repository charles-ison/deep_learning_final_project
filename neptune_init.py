import configparser
import neptune
from neptune.integrations.pytorch import NeptuneLogger

# print("Neptune imported.")
def _process_api_key(f_key: str) -> configparser.ConfigParser:
    api_key = configparser.ConfigParser()
    api_key.read(f_key)
    return api_key

def init_neptune(cfg: str):
    # You will need to store your neptune project id and api key
    # in an external file. Please do not hard-code these values - 
    # it is a security risk.
    # Do not commit this credentials file to your github repository.
    creds = _process_api_key(cfg)
    runtime = neptune.init_run(project=creds['CLIENT_INFO']['project_id'],
                        api_token=creds['CLIENT_INFO']['api_token'])
    return runtime

credentials = 'credentials/neptune.ini'

runtime = init_neptune(credentials)