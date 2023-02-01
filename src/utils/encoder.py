import torch
import json
import numpy as np
from ase.spacegroup import Spacegroup

__all__ = ["CrystalEncoder"]


class CrystalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, Spacegroup):
            return {"number": obj.no, "symbol": obj.symbol}
        # if isinstance(obj, tf.Tensor):
        #    return obj.numpy().tolist()
        return json.JSONEncoder.default(self, obj)
