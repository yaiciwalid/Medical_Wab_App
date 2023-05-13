'Application configuration file'
import json
import os


class DataConfig:
    """ Class to read configuration parameters from files located in ./configuration"""
    _input_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    _infra_filepath = os.path.join(_input_path, "configuration", "config.json")

    def __init__(self):
        """ Initialize data configuration"""
        self._prepare_infra_configuration()

    def _prepare_infra_configuration(self):
        """Load infra configuration"""
        with open(self._infra_filepath) as json_file:
            self.infra_config = json.load(json_file)
