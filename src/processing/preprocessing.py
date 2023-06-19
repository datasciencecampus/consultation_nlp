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
    if type(filepath) is not str: 
        raise TypeError("filepath must be a string")
    
    with open(filepath) as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exception:
            print(exception)
    