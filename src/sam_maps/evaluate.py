import hydra
from omegaconf import OmegaConf

from sam_maps.models import build_model


@hydra.main(version_base=None, config_path="configs", config_name="sam-maps")
def main(config):
    # set_seed(config.seed)
    OmegaConf.set_struct(config, False)  # Open the struct
    config = OmegaConf.merge(config, config.method)
    config["eval"] = True

    # Load the model
    model = build_model(config)
    print(model)


if __name__ == "__main__":
    main()
