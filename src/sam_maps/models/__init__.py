from sam_maps.models.model import Model

from sam_maps.models.RGE.samroad import SAMRoad

from sam_maps.models.RS.auto_rs import AutoRS

from sam_maps.models.RC.auto_rc import AutoRC


rge_modules = {
    "samroad": SAMRoad,
}
rs_modules = {
    "auto-rs": AutoRS,
}
rc_modules = {
    "auto-rc": AutoRC,
}


def build_model(config):
    rge = rge_modules[config.method.model_name](config=config)
    rs = rs_modules[config.method.model_name](config=config)
    rc = rc_modules[config.method.model_name](config=config)

    model = Model(config, rge, rs, rc)

    return model


def load_model(config):
    raise NotImplementedError
    model = build_model(config).load_from_checkpoint(config.ckpt_path)  # TODO
    return model
