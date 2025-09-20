import unittest
import torch
from torch.testing._internal.common_utils import TestCase
from torch.nn.utils.rnn import PackedSequence, pack_sequence

try:
    from src.models import LSTMNet
except:
    import sys
    import os.path as osp
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-3])
    sys.path.append(dir_path)
    from src.models import LSTMNet


class TestLSTMNet(TestCase):
    __expected = torch.tensor([
        [0.12417398393154144],
        [0.12434756755828857],
        [0.12447639554738998],
        [0.12432965636253357],
        [0.12444637715816498],
        [0.12428965419530869],
        [0.12438388168811798],
        [0.12427278608083725],
        [0.12427915632724762],
        [0.12439240515232086],
        [0.12417805939912796],
        [0.12433996796607971],
        [0.12433985620737076],
        [0.1244719922542572],
        [0.1241849958896637],
        [0.12435460090637207]]
    )
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        self.model_bce = LSTMNet(
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