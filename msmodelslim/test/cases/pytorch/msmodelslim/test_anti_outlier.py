import torch

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

setattr(AntiOutlier, 'init_dag', lambda *args, **kargs: None)
setattr(AntiOutlier, '_process', lambda *args, **kargs: None)


class OneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.l1 = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = self.l1(x)

        return x


def test_anti_outlier_m1():
    model = OneModel()

    dataset_calib = [[torch.FloatTensor(8, 8)]]
    anti_config = AntiOutlierConfig(anti_method="m1")
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()


def test_anti_outlier_m2():
    model = OneModel()

    dataset_calib = [[torch.FloatTensor(8, 8)]]
    anti_config = AntiOutlierConfig(anti_method="m2")
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()


def test_anti_outlier_m3():
    model = OneModel()

    dataset_calib = [[torch.FloatTensor(8, 8)]]
    anti_config = AntiOutlierConfig(w_bit=8, a_bit=16, anti_method="m3")
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()