import unittest
import torch
from torch.testing._internal.common_utils import TestCase

try:
    from src.models import GRUNet
except:
    import sys
    import os.path as osp
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-3])
    sys.path.append(dir_path)
    from src.models import GRUNet


class TestLSTMNet(TestCase):
    __expected = torch.tensor([
        [-0.13184911012649536],
        [-0.13030317425727844],
        [-0.13904204964637756],
        [-0.13663940131664276],
        [-0.13334624469280243],
        [-0.13007816672325134],
        [-0.1424441933631897],
        [-0.1362936794757843],
        [-0.13395407795906067],
        [-0.12812957167625427],
        [-0.12789931893348694],
        [-0.13738679885864258],
        [-0.14499229192733765],
        [-0.14220893383026123],
        [-0.12532997131347656],
        [-0.13381163775920868]]
    )
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        self.model_bce = GRUNet(
            input_size=8,
            hidden_size=16, 
            num_layers=4,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
            #
            task='classification',
            d_out=1,
            d_hidden=None,
            activation=None,
            reduced=True,
            cls_method='autoregressive',
            loss_fn='bce',
        )

        #self.model_ce = LSTMNet(
        #    input_size=8,
        #    hidden_size=16, 
        #    num_layers=4,
        #    bias=True,
        #    batch_first=False,
        #    dropout=0.0,
        #    bidirectional=False,
        #    #
        #    task='classification',
        #    d_out=4,
        #    d_hidden=None,
        #    activation=None,
        #    reduced=True,
        #    cls_method='autoregressive',
        #    loss_fn='ce',
        #)

    
    def sample_inputs(self, device: torch.device):
        return torch.load('test/models/sample_batch.pt').to(device)

    def _test_correctness(self, device):
        batch_size=16
        batch = self.sample_inputs(device=device)
        out_bce = self.model_bce(batch, batch_size=batch_size)
        torch.testing.assert_close(out_bce, self.__expected)
        
    def test_correctness_cpu(self):
        self._test_correctness("cpu")
    
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")


if __name__ == "__main__":
    unittest.main()