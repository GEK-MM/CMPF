"""CMPF 最终模型：``CMPFMainBranch`` + ``CMPFGatedHead``。"""

from model.cmpf_model import CMPFModel
from model.cmpf_decoder import CMPFGatedHead


class CMPFGatedModel(CMPFModel):
    """保持 ``CMPFMainBranch`` 不变，仅替换为新增门控解码器。"""

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
        self.decoder = CMPFGatedHead(
            target_size=(img_size, img_size),
            dropout=drop,
            gate_dropout=drop_act,
        )
