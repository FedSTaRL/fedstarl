import unittest
import torch
from torch.testing._internal.common_utils import TestCase
from torch.nn.utils.rnn import PackedSequence, pack_sequence

try:
    from src.models import RNNNet
except:
    import sys
    import os.path as osp
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-3])
    sys.path.append(dir_path)
    from src.models import RNNNet


class TestLSTMNet(TestCase):
    __expected = torch.tensor([
        [0.40332716703414917],
        [0.3853909373283386],
        [0.38881808519363403],
        [0.4005470275878906],
        [0.3714591860771179],
        [0.41226500272750854],
        [0.3754558563232422],
        [0.3815951347351074],
        [0.37834632396698],
        [0.3869065046310425],
        [0.3769581913948059],
        [0.4013410210609436],
        [0.4083407521247864],
        [0.3735920190811157],
        [0.38838499784469604],
        [0.40669533610343933]]
    )
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        self.model_bce = RNNNet(
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