import timm

import argus


class RanzcrModel(argus.Model):
    nn_module = {
        "timm": timm.create_model
    }
