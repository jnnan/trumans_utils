import torch
import smplx
from constants import *
from scipy.interpolate import interp1d
from torch import nn, einsum
from pytorch3d import transforms as T


class JointsToSMPLX(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def optimize_smpl(pose_pred, joints, joints_ind, hand_pca=45):
    device = joints.device
    len = joints.shape[0]

    smpl_model = smplx.create(SMPL_DIR, model_type='smplx',
                              gender='male', ext='npz',
                              num_betas=10,
                              use_pca=False,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=len,
                              ).to(device)
    smpl_model.eval()

    # weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 100]).reshape(nb_joints, 1).repeat(1, 3).to(device)
    joints = joints.reshape(len, -1, 3) + torch.tensor(pelvis_shift).to(device)
    pose_input = torch.nn.Parameter(pose_pred.detach(), requires_grad=True)
    transl = torch.nn.Parameter(torch.zeros(pose_pred.shape[0], 3).to(device), requires_grad=True)
    # left_hand = torch.nn.Parameter(torch.zeros(pose_pred.shape[0], hand_pca).to(device), requires_grad=True)
    # right_hand = torch.nn.Parameter(torch.zeros(pose_pred.shape[0], hand_pca).to(device), requires_grad=True)
    left_hand = torch.from_numpy(relaxed_hand_pose[:45].reshape(1, -1).repeat(pose_pred.shape[0], axis=0)).to(device)
    right_hand = torch.from_numpy(relaxed_hand_pose[45:].reshape(1, -1).repeat(pose_pred.shape[0], axis=0)).to(device)
    optimizer = torch.optim.Adam(params=[pose_input, transl], lr=0.05)
    loss_fn = nn.MSELoss()
    for step in range(100):
        smpl_output = smpl_model(transl=transl, body_pose=pose_input[:, 3:], global_orient=pose_input[:, :3], return_verts=True,
                                 left_hand_pose=left_hand,# @ left_hand_components[:hand_pca],
                                 right_hand_pose=right_hand,# @ right_hand_components[:hand_pca],
                                 )
        joints_output = smpl_output.joints[:, joints_ind].reshape(len, -1, 3)
        loss = loss_fn(joints[:, :], joints_output[:, :])
        # loss = torch.mean((joints - joints_output) ** 2 * weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())



    #left_hand = left_hand @ left_hand_components[:hand_pca]
    #right_hand = right_hand @ right_hand_components[:hand_pca]

    return pose_input.detach().cpu().numpy(), transl.detach().cpu().numpy(), left_hand.detach().cpu().numpy(), right_hand.detach().cpu().numpy()


def joints_to_smpl(model, joints, joints_ind, interp_s):
    joints = interpolate_joints(joints, scale=interp_s)
    # joints = interpolate_joints(joints, scale=0.33)
    # joints = interpolate_joints(joints, scale=interp_s * 3)
    input_len = joints.shape[0]
    joints = joints.reshape(input_len, -1, 3)
    joints = joints.permute(1, 0, 2)
    trans_np = joints[0].detach().cpu().numpy()
    joints = joints - joints[0]
    joints = joints.permute(1, 0, 2)
    joints = joints.reshape(input_len, -1)
    pose_pred = model(joints)
    pose_pred = pose_pred.reshape(-1, 6)
    pose_pred = T.matrix_to_axis_angle(T.rotation_6d_to_matrix(pose_pred)).reshape(input_len, -1)
    # pose_pred = pose_pred[:seq_len]
    pose_output, transl, left_hand, right_hand = optimize_smpl(pose_pred, joints, joints_ind)

    transl = trans_np - np.array(pelvis_shift) + transl
    return pose_output, transl, left_hand, right_hand


def interpolate_joints(joints, scale):
    if scale == 1:
        return joints
    device = joints.device
    joints = joints.detach().cpu().numpy()
    in_len = joints.shape[0]
    out_len = int(in_len * scale)
    joints = joints.reshape(in_len, -1)
    x = np.array(range(in_len))
    xnew = np.linspace(0, in_len - 1, out_len)
    f = interp1d(x, joints, axis=0)
    joints_new = f(xnew)
    joints_new = torch.from_numpy(joints_new).to(device).float()

    return  joints_new