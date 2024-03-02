import platform
import torch
from modules import devices


def test(device: torch.device):
    try:
        ten1 = torch.randn((2, 4,), device=device)
        ten2 = torch.randn((4, 8,), device=device)
        out = torch.mm(ten1, ten2)
        return out.sum().is_nonzero()
    except Exception:
        return False


def initialize_zluda():
    device = devices.get_optimal_device()
    if platform.system() == "Windows" and torch.cuda.is_available() and torch.cuda.get_device_name(device).endswith("[ZLUDA]"):
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        devices.device_codeformer = devices.cpu

        if not test(device):
            print(f'ZLUDA device failed to pass basic operation test: index={device.index}, device_name={torch.cuda.get_device_name(device)}')
            torch.cuda.is_available = lambda: False
            devices.backend = 'cpu'
            devices.device = devices.device_esrgan = devices.device_gfpgan = devices.device_interrogate = devices.cpu
