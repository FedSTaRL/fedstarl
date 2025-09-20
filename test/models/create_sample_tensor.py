import torch
from torch.nn.utils.rnn import pack_sequence

def main():
    t = [torch.rand(torch.randint(10, 256, (1,)).item(), 8) for _ in range(16)]
    sample_tensor = pack_sequence(t, enforce_sorted=False)
    torch.save(sample_tensor, "../sample_tensor.pt")


if __name__ == "__main__":
    main()

