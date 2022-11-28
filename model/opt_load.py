import torch
from model.sam import SAM


def opt_load(model, config):
    if config["MODEL"]["OPTIMIZER"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["MODEL"]["LR"],
            weight_decay=config["MODEL"]["WEIGHT_DECAY"],
        )
    elif config["MODEL"]["OPTIMIZER"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["MODEL"]["LR"],
            weight_decay=config["MODEL"]["WEIGHT_DECAY"],
        )
    elif config["MODEL"]["OPTIMIZER"] == "SGD-Nesterov":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["MODEL"]["LR"],
            weight_decay=config["MODEL"]["WEIGHT_DECAY"],
            momentum=config["MODEL"]["MOMENTUM"],
            nesterov=True,
        )
    elif config["MODEL"]["OPTIMIZER"] == "SAM":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=config["MODEL"]["LR"],
            momentum=config["MODEL"]["MOMENTUM"],
            nesterov=True,
        )
    else:
        assert False, "No Optimizer Setting"

    return optimizer, config["MODEL"]["OPTIMIZER"]
