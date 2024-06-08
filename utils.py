import torch
import hydra
import numpy as np
from einops import rearrange
import random
import os


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def transform_points(x, mat):
    shape = x.shape
    x = rearrange(x, 'b t (j c) -> b (t j) c', c=3)  # B x N x 3
    x = torch.einsum('bpc,bck->bpk', mat[:, :3, :3], x.permute(0, 2, 1))  # B x 3 x N   N x B x 3
    x = x.permute(2, 0, 1) + mat[:, :3, 3]
    x = x.permute(1, 0, 2)
    x = x.reshape(shape)

    return x


def create_meshgrid(bbox, size, batch_size=1):
    x = torch.linspace(bbox[0], bbox[1], size[0])
    y = torch.linspace(bbox[2], bbox[3], size[1])
    z = torch.linspace(bbox[4], bbox[5], size[2])
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    grid = grid.repeat(batch_size, 1, 1)

    # aug_z = 0.75 + torch.rand(batch_size, 1) * 0.35
    # grid[:, :, 2] = grid[:, :, 2] * aug_z

    return grid


def zup_to_yup(coord):
    # change the coordinate from yup to zup
    if len(coord.shape) > 1:
        coord = coord[..., [0, 2, 1]]
        coord[..., 2] *= -1
    else:
        coord = coord[[0, 2, 1]]
        coord[2] *= -1

    return coord


def rigid_transform_3D(A, B, scale=False):
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        # return None, None, None
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T

    return c, R, t


def find_free_port():
    from contextlib import closing
    import socket

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def init_model(model_cfg, device, eval, load_state_dict=False):
    model = hydra.utils.instantiate(model_cfg)
    if eval:
        load_state_dict_eval(model, model_cfg.ckpt, device=device)
    else:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False,
                                                          find_unused_parameters=True)
        if load_state_dict:
            model.module.load_state_dict(torch.load(model_cfg.ckpt))
            model.train()

    return model


def load_state_dict_eval(model, state_dict_path, map_location='cuda:0', device='cuda'):
    state_dict = torch.load(state_dict_path, map_location=map_location)
    key_list = [key for key in state_dict.keys()]
    for old_key in key_list:
        new_key = old_key.replace('module.', '')
        state_dict[new_key] = state_dict.pop(old_key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


class dotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args):
        val = dict.get(*args)
        return dotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__