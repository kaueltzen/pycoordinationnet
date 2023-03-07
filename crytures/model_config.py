

## ----------------------------------------------------------------------------

_model_config = {
    'composition'           : False,
    'sites'                 : False,
    'site_features'         : False,
    'site_features_ces'     : False,
    'site_features_oxid'    : False,
    'site_features_csms'    : False,
    'site_features_ligands' : False,
    'ligands'               : False,
    'ce_neighbors'          : False,
}

## ----------------------------------------------------------------------------

class ModelConfig(dict):
    def __init__(self, *args, **kwargs):
        # Set default values
        for key, value in _model_config.items():
            self[key] = value
        # Check that we do not accept any invalid
        # config options
        for key, value in kwargs.items():
            if key not in self:
                raise KeyError(key)
        # Override default config values
        super().__init__(self, *args, **kwargs)

    def __setitem__(self, key, value):
        if key in _model_config:
            super().__setitem__(key, value)
        else:
            raise KeyError(key)

    def __str__(self):
        result = 'Model config:\n'
        for key, value in self.items():
            result += f'-> {key:21}: {value}\n'
        return result

## ----------------------------------------------------------------------------

DefaultModelConfig = ModelConfig(
    site_features      = True,
    site_features_ces  = True,
    site_features_oxid = True,
    site_features_csms = True)