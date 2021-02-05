import torch.nn as nn
import torch.nn.functional as F

from timm import create_model

from src.models.nn_modules import SpatialAttention


class TimmModel(nn.Module):
    def __init__(self,
                 model_name="tf_efficientnet_b3_ns",
                 attention=None,
                 **kwargs):
        super().__init__()
        self.model = create_model(model_name, **kwargs)
        if attention is not None:
            self.attention = SpatialAttention(**attention)
        else:
            self.attention = None
        self.model.classifier = self.model.get_classifier()

    def forward(self, x, features=False, attention=False):
        output = []
        feat = self.model.forward_features(x)
        if features:
            output.append(feat)
        if self.attention is not None:
            att = self.attention(feat)
            if attention:
                output.append(att)
            feat = att * feat
        x = self.model.global_pool(feat)
        if self.model.drop_rate > 0.:
            x = F.dropout(x, p=self.model.drop_rate, training=self.training)
        x = self.model.classifier(x)

        if not len(output):
            return x
        return tuple([x] + output)
