import torch

def test_cuda():
    if torch.cuda.is_available():
        print("Detected Cuda Cores, setting Device to Cuda")
        dev = "cuda:0"
    else:
        print("No cuda cores detected, using CPU")
        dev = "cpu"
    return dev
