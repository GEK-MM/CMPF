"""使用门控解码器的完整表达式耦合模型。"""

from model.coupled_model import CoupledModel
from model.gated_scale_adaptive_decoder import GatedScaleAdaptiveHead


class CoupledGatedDecoderModel(CoupledModel):
    """保持 ``CoupledMainBranch`` 不变，仅替换为新增门控解码器。"""

    def __init__(self, tunelang=False, num_query=20, length=17,
                 img_size=406, text_model_path='', drop=0.1, drop_act=0.0):
        super().__init__(
            tunelang=tunelang,
            num_query=num_query,
            length=length,
            img_size=img_size,
            text_model_path=text_model_path,
            drop=drop,
            drop_act=drop_act,
        )
        self.decoder = GatedScaleAdaptiveHead(
            target_size=(img_size, img_size),
            dropout=drop,
            gate_dropout=drop_act,
        )
