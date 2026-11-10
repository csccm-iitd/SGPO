"""Pre-trained U-FNO (Wen et al. 2022) as the LoGoS-GPO mean operator for the CO2 storage benchmark.

U-FNO code and checkpoints are not part of this repository: get them from the U-FNO authors
(https://github.com/gegewen/ufno; license CC BY-NC-ND, not redistributed here) and point to the clone with
--ufno-dir or the environment variable UFNO_DIR (the directory that contains ufno.py).

LoGoS-GPO needs features(x) -> per-point hidden features h and proj(h) -> mean. For U-FNO, h is the output of its
first projection layer (fc1 + ReLU, 128 channels at every (z, r, t)) and proj is its last layer (fc2), so the
predictive mean is exactly the pre-trained model. Inputs are the (b, 96, 200, 11) static channels of `co2_sg`; the
24-step time channel is appended to rebuild U-FNO's (b, 96, 200, 24, 12) input.

    UFNO_DIR=/path/to/ufno python experiments/ufno_mean.py sg_UFNO_4500_trained.pt --out runs/co2_ufno/op
    UFNO_DIR=/path/to/ufno python experiments/train_logos_vnngp.py --init runs/co2_ufno/op --out runs/co2_ufno/logos ...
The first command writes a stage-1 style ckpt.pt (and, with --test-pred, test predictions for co2_eval.py --pred).
"""
import argparse, importlib, os, pickle, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.nn as nn
import torch.nn.functional as F
from sgpo.data.benchmarks import RAW


def import_ufno(ufno_dir=None):
    """Imports ufno.py from a user-provided U-FNO clone (--ufno-dir or UFNO_DIR)."""
    path = ufno_dir or os.environ.get("UFNO_DIR")
    if path:
        sys.path.insert(0, os.path.abspath(path))
    try:
        return importlib.import_module("ufno")
    except ImportError as e:
        raise ImportError("U-FNO code not found: clone https://github.com/gegewen/ufno and set UFNO_DIR "
                          "(or --ufno-dir) to the directory containing ufno.py") from e


def time_channel():
    """The 24 snapshot times of the U-FNO data (input channel 11), cached next to the data."""
    cache = f"{RAW}/ufno/time_channel.pt"
    if not os.path.exists(cache):
        a = torch.load(f"{RAW}/ufno/sg_test_a.pt", mmap=True, weights_only=True)
        torch.save(a[0, 0, 0, :, 11].clone(), cache)
    return torch.load(cache, weights_only=True).float()


class UFNOMean(nn.Module):
    def __init__(self, modes=(10, 10, 10), width=36, ufno_dir=None):
        super().__init__()
        self.net = import_ufno(ufno_dir).Net3d(*modes, width)
        self.register_buffer("times", time_channel())

    def features(self, x):
        b, nz, nr, _ = x.shape
        nt = len(self.times)
        x5 = torch.cat([x[:, :, :, None, :].expand(b, nz, nr, nt, x.shape[-1]),
                        self.times.view(1, 1, 1, nt, 1).expand(b, nz, nr, nt, 1)], -1)
        x5 = F.pad(F.pad(x5, (0, 0, 0, 8, 0, 8), "replicate"), (0, 0, 0, 0, 0, 0, 0, 8), "constant", 0)  # as Net3d
        blk = self.net.conv1
        fc2 = blk.fc2
        blk.fc2 = nn.Identity()
        try:
            h = blk(x5)                                    # SimpleBlock3d up to fc1 + ReLU (fc2 bypassed)
        finally:
            blk.fc2 = fc2
        return h[:, :nz, :nr, :nt]

    def proj(self, h):
        return self.net.conv1.fc2(h)

    def forward(self, x):
        return self.proj(self.features(x))[..., 0]


# The published checkpoint is a pickled Net3d. It is loaded with an allow-list unpickler: only the U-FNO classes and
# the torch layers they use can be constructed, never arbitrary objects.
TORCH_OK = {("torch.nn.modules.linear", "Linear"), ("torch.nn.modules.conv", "Conv1d"), ("torch.nn.modules.conv", "Conv3d"),
            ("torch.nn.modules.conv", "ConvTranspose3d"), ("torch.nn.modules.container", "Sequential"),
            ("torch.nn.modules.batchnorm", "BatchNorm3d"), ("torch.nn.modules.activation", "LeakyReLU"),
            ("torch.nn.modules.dropout", "Dropout"), ("torch._utils", "_rebuild_parameter"),
            ("torch._utils", "_rebuild_tensor_v2"), ("collections", "OrderedDict"), ("torch", "FloatStorage"),
            ("torch", "ComplexFloatStorage"), ("torch", "LongStorage")}
UFNO_OK = {"Net3d", "SimpleBlock3d", "SpectralConv3d", "U_net"}


def safe_pickle(ufno):
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in ("__builtin__", "builtins") and name == "set":
                return set
            if module == "__main__" and name in UFNO_OK:
                return getattr(ufno, name)
            if (module, name) in TORCH_OK:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"refused {module}.{name}")
    return types.SimpleNamespace(Unpickler=Unpickler, load=lambda f, **kw: Unpickler(f, **kw).load(), __name__="safe_pickle")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="pre-trained U-FNO checkpoint (e.g. sg_UFNO_4500_trained.pt)")
    p.add_argument("--out", required=True, help="output dir for the stage-1 style ckpt.pt")
    p.add_argument("--ufno-dir", default=None, help="U-FNO clone containing ufno.py (default: $UFNO_DIR)")
    p.add_argument("--test-pred", default=None, help="also write the test-set predictions (500, 96, 200, 24) to this file")
    p.add_argument("--batch", type=int, default=2)
    a = p.parse_args()
    ufno = import_ufno(a.ufno_dir)
    net = torch.load(a.ckpt, map_location="cpu", weights_only=False, pickle_module=safe_pickle(ufno))
    blk = net.conv1
    modes = (blk.conv0.modes1, blk.conv0.modes2, blk.conv0.modes3)
    m = UFNOMean(modes, blk.width)
    m.net.load_state_dict(net.state_dict())
    os.makedirs(a.out, exist_ok=True)
    A = {"arch": "ufno", "modes": list(modes), "width": blk.fc1.out_features, "ufno_width": blk.width, "data": "co2_sg",
         "ntrain": 4500, "nval": 500, "ntest": 500, "tail_test": False, "out_ch": 24, "mask_ch": 0, "pad": 0}
    z = torch.zeros(())
    torch.save({"args": A, "state": m.state_dict(), "xm": z, "xs": z + 1, "ym": z, "ys": z + 1}, f"{a.out}/ckpt.pt")
    print("wrote", f"{a.out}/ckpt.pt", A)
    if a.test_pred:
        net = net.cuda().eval()
        xa = torch.load(f"{RAW}/ufno/sg_test_a.pt", mmap=True, weights_only=True)    # (500, 96, 200, 24, 12)
        with torch.no_grad():
            out = torch.cat([net(xa[i:i + a.batch].cuda()).view(-1, 96, 200, 24).cpu() for i in range(0, len(xa), a.batch)])
        os.makedirs(os.path.dirname(os.path.abspath(a.test_pred)), exist_ok=True)
        torch.save(out, a.test_pred)
        print("wrote", a.test_pred, tuple(out.shape))
