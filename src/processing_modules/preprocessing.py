import yaml

def load_config(filepath:str):
    ''' Loads configuration settings from given filepath to
    yaml file

    Parameters
    ----------
    filepath : str
        The relative filepath to the yaml file

    Returns
    -------
    dict
        the configuration settings with key-value pairs
    '''
    assert type(filepath) is str, "filepath must be a string"
    
    with open(filepath) as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)
    return config