from utils import registry

#print(3)
MODELS = registry.Registry('models')
#print('models',MODELS)

def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    #print("build")
    return MODELS.build(cfg, **kwargs)
#print(5)

