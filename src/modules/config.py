import json

import yaml
from schema import And, Or, Schema, SchemaError


class Config:
    """configuration settings for the system with user input validation"""

    def __init__(self):
        config = {}
        config["general"] = self._load_config("src/general.yaml")
        config["spelling"] = self._load_json("src/spelling.json")
        config["models"] = self._load_config("src/models.yaml")
        self.valid_config = self._check(config)
        self.settings = config

    def _check(self, config):
        """Check that config is not throwing any errors

        Parameters
        ----------
        config:dict
            configuration settings for the system

        Returns
        -------
        bool
            True if there are no errors in the configuration file

        Raises
        ------
        SchemaError
            Error describing what is wrong with the config file
        """
        try:
            self._validate_config(config)
            return True
        except SchemaError as exception:
            print(exception)
            return False

    def _validate_config(self, config: dict) -> None:
        """validates configuration file according to pre-defined schema

        Parameters
        ----------
        config:dict
            a dictionary containing the configuration settings

        Raises
        ------
        SchemaError
            If configuration file does not match the expected schema

        Returns
        -------
        None
        """
        schema = Schema(
            {
                "general": {
                    "raw_data_path": str,
                    "additional_stopwords": list,
                    "lemmatize": bool,
                },
                "models": {
                    str: {
                        "max_features": Or(And(int, self._greater_than_zero), None),
                        "ngram_range": (
                            And(int, self._greater_than_zero),
                            And(int, self._greater_than_zero),
                        ),
                        "min_df": Or(
                            And(float, self._between_zero_and_one),
                            And(int, self._greater_than_zero),
                        ),
                        "max_df": Or(
                            And(float, self._between_zero_and_one),
                            And(int, self._greater_than_zero),
                        ),
                        "n_topics": And(int, self._greater_than_zero),
                        "n_top_words": And(int, self._greater_than_zero),
                        "max_iter": {
                            "lda": And(int, self._greater_than_zero),
                            "nmf": And(int, self._greater_than_zero),
                        },
                        "lowercase": bool,
                        "topic_labels": {
                            "lda": Or(None, [str]),
                            "nmf": Or(None, [str]),
                        },
                    }
                },
                "spelling": {str: int},
            }
        )

        schema.validate(config)
        return None

    def _greater_than_zero(self, n: int):
        """function to check if n is greater than zero

        Parameters
        ----------
        n:int
            a numeric value to check

        Returns
        -------
        bool
            True if n is greater than zero
        """
        return n > 0

    def _between_zero_and_one(self, n: float):
        """function to check if n is between zero and one

        Parameters
        ----------
        n:float
            a numeric value to check

        Returns
        -------
        bool
            True if n is between zero and one"""
        return 0.0 <= n <= 1.0

    def _load_config(self, filepath: str) -> dict:
        """Loads configuration settings from given filepath to
        yaml file

        Parameters
        ----------
        filepath : str
            The relative filepath to the yaml file

        Returns
        -------
        dict
            the configuration settings with key-value pairs
        """
        if type(filepath) is not str:
            raise TypeError("filepath must be a string")

        with open(filepath, "r") as file:
            config = yaml.load(file, Loader=yaml.Loader)
        return config

    def _load_json(self, filepath: str) -> dict:
        """Loads json file as dictionary
        Parameters
        ----------
        filepath:str
            the filepath to where the json file is stored
        Returns
        -------
        dict
            the json file in dict format
        """
        if type(filepath) is not str:
            raise TypeError("filepath must be a string")
        with open(filepath, "r") as file:
            json_data = json.load(file)
        return json_data
