import os
import pdb
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
# from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
from models.joints_to_smplx import joints_to_smpl, JointsToSMPLX
from utils import *
from constants import *
from datasets.trumans import TrumansDataset
from models.synhsi import Unet
# from hydra import compose, initialize
import yaml


ACT_TYPE = ['scene', 'grasp', 'artic', 'none']


def convert_trajectory(trajectory):
    trajectory_new = [[t['x'], t['y'], t['z']] for t in trajectory]
    trajectory_new = np.array(trajectory_new)
    return trajectory_new


def get_base_speed(cfg, trajectory, is_zup=True):
    trajectory_layer = trajectory
    if is_zup:
        trajectory_layer = zup_to_yup(trajectory_layer)
    trajectory2D = trajectory_layer[:, [0, 2]]
    distance = np.sum(np.linalg.norm(trajectory2D[1:] - trajectory2D[:-1], axis=1))
    speed = trajectory_layer.shape[0] // distance
    print('Base Speed:', speed, flush=True)

    return speed


def get_guidance(cfg, trajectory, samplers, act_type='none', speed=35):
    trajectory_layer = trajectory
    # trajectory_layer = zup_to_yup(trajectory_layer)
    print(trajectory_layer.shape)
    if cfg.action_type != 'pure_inter':
        #TODO
        midpoints = trajectory_layer[[0] * cfg.len_pre + list(range(0, len(trajectory_layer), speed)) + [-1] * (cfg.len_act + (1 if cfg.stay_and_act else 0))]
        # midpoints[0, 0] = 1.4164
        # midpoints[0, 2] = 2.2544
        # midpoints = trajectory_layer[[0] * cfg.len_pre + [25, 50] + [50] + [70, 90]]

    else:
        midpoints = trajectory_layer[[0] * cfg.len_pre + [0] + [0] * (cfg.len_act + 1 if cfg.stay_and_act else 0)]
    midpoints = torch.tensor(midpoints).float().to(cfg.device)
    max_step = midpoints.shape[0] - 1

    mat_init = cfg.batch_size * [np.eye(4)]
    mat_init = torch.from_numpy(np.stack(mat_init, axis=0)).float().to(cfg.device)
    print(midpoints)
    mat_init[:, 0, 3] = midpoints[0, 0]
    mat_init[:, 2, 3] = midpoints[0, 2]

    dx = midpoints[cfg.len_pre + 1, 0] - midpoints[0, 0]
    dz = midpoints[cfg.len_pre + 1, 2] - midpoints[0, 2]

    print(-np.arctan2(dx.item(), dz.item()), dx, dz)

    mat_rot_y = R.from_rotvec(np.array([0, np.arctan2(dx.item(), dz.item()), 0])).as_matrix()
    # mat_rot_y = R.from_rotvec(np.array([0, np.arctan2(-1, -2), 0])).as_matrix()
    mat_init[:, :3, :3] = torch.from_numpy(mat_rot_y).float().to(cfg.device)


    # goal_list = torch.zeros((max_step, cfg.batch_size, cfg.dataset.seq_len, 3)).float().to(cfg.device)
    goal_list = []
    action_label_list = []
    # action_label_list = torch.zeros((max_step, cfg.batch_size, cfg.dataset.seq_len, cfg.dataset.nb_actions)).float().to(cfg.device)
    sampler_list = []

    if act_type == 'none':
        for s in range(max_step):
            goal = torch.zeros((cfg.batch_size, 1, 3)).float().to(cfg.device)
            goal[:, :] = midpoints[s + 1]
            goal_list.append(goal)
            if cfg.dataset.nb_actions > 0:
                action_label = torch.zeros((cfg.batch_size, cfg.dataset.seq_len, cfg.dataset.nb_actions)).float().to(cfg.device)
                action_label_list.append(action_label)
            else:
                action_label_list.append(None)
            sampler_list.append(samplers['body'])
    elif act_type == 'write':
        midpoints = torch.from_numpy(trajectory_layer[::4]).float().to(cfg.device)
        for s in range(midpoints.shape[0] // 16):
            goal = torch.zeros((cfg.batch_size, 16, 3)).float().to(cfg.device)
            goal[:, :] = midpoints[s * 16: (s + 1) * 16]
            goal_list.append(goal)
            if cfg.dataset.nb_actions > 0:
                action_label = torch.zeros((cfg.batch_size, cfg.dataset.seq_len, cfg.dataset.nb_actions)).float().to(cfg.device)
                action_label_list.append(action_label)
            else:
                action_label_list.append(None)
            sampler_list.append(samplers['hand'])
    elif act_type == 'scene':
        for s in range(max_step):
            goal = torch.zeros((cfg.batch_size, 1, 3)).float().to(cfg.device)
            goal[:, :] = midpoints[s + 1]
            goal_list.append(goal)
            sampler_list.append(samplers['body'])
            action_label = torch.zeros((cfg.batch_size, cfg.dataset.seq_len, cfg.dataset.nb_actions)).float().to(cfg.device)
            if s > max_step - cfg.len_act:
                action_label[:, :, cfg.action_id] = 1.
            action_label_list.append(action_label)
    elif act_type == 'grasp':
        for s in range(max_step):
            if s != max_step - cfg.len_act:
                goal = torch.zeros((cfg.batch_size, 1, 3)).float().to(cfg.device)
                goal[:, :] = midpoints[s + 1]
                goal_list.append(goal)
            else:
                # grasp_goal = zup_to_yup(np.array([[-0.32, -3.36, 0.395],
                #                                   [-0.3, -3.2, 0.395],
                #                                   [-0.25, -3, 0.394],
                #                                   [-0.147, -3.0, 0.395]])).reshape((cfg.batch_size, -1, 3))

                grasp_goal = zup_to_yup(np.array(trajectory['Object'])).reshape((cfg.batch_size, 1, 3))
                goal = torch.zeros((cfg.batch_size, 3, 3)).float().to(cfg.device)
                goal[:, :] = torch.from_numpy(grasp_goal).float().to(cfg.device)
                goal_list.append(goal)

            action_label = torch.zeros((cfg.batch_size, cfg.dataset.seq_len, cfg.dataset.nb_actions)).float().to(cfg.device)
            if s < max_step - cfg.len_act:
                sampler_list.append(samplers['body'])
            elif s == max_step - cfg.len_act:
                sampler_list.append(samplers['hand'])
            else:
                sampler_list.append(samplers['body'])
                if cfg.action_id != -1:
                    action_label[:, :, cfg.action_id] = 1.
            action_label_list.append(action_label)
    elif act_type == 'pure_inter':
        sampler_list += [samplers['body']] * (cfg.len_pre + cfg.len_act)
        goal = torch.zeros((cfg.batch_size, 1, 3)).float().to(cfg.device)
        goal[:, :] = midpoints[0]
        goal_list += [goal] * (cfg.len_pre + + cfg.len_act)
        action_label = torch.zeros((cfg.batch_size, cfg.dataset.seq_len, cfg.dataset.nb_actions)).float().to(cfg.device)
        action_label_list += [action_label.clone()] * cfg.len_pre
        action_label[:, :, cfg.action_id] = 2.
        action_label_list += [action_label.clone()] * cfg.len_act


    return mat_init, goal_list, action_label_list, sampler_list


def sample_step(cfg, mat, obj_locs, goal_list, action_label_list, sampler_list):
    max_step = len(goal_list)
    fixed_points = None
    fixed_frame = 2
    points_all = []
    cnt_fixed_frame = 0
    cnt_seq_len = 0


    for s in range(max_step):
        print('step', s)

        sampler = sampler_list[s]
        if s != 0:
            fixed_points = sampler.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))
        else:
            if cfg.continue_last:
                method_id = cfg.method_name.split('_')[-1]
                method_name_last = cfg.method_name[:-1] + str(int(method_id) - 1)
                mat = torch.from_numpy(np.load(os.path.join(cfg.exp_dir, f'{method_name_last}_mat.npy'))).to(sampler.device)
                fixed_points = torch.from_numpy(np.load(os.path.join(cfg.exp_dir, f'{method_name_last}_fixed_points.npy'))).to(sampler.device)
                fixed_points = sampler.dataset.normalize_torch(transform_points(fixed_points, torch.inverse(mat)))

        samples, occs = sampler.p_sample_loop(fixed_points, obj_locs, mat, cfg.scene_name, goal_list[s], action_label_list[s])

        if 0 <= s < cfg.len_pre:
            cnt_fixed_frame += sampler.fixed_frame
        if 0 <= s < cfg.len_pre:
            cnt_seq_len += cfg.dataset.seq_len

        points_gene = samples[-1]
        points_gene_np = points_gene.reshape(cfg.batch_size, cfg.dataset.seq_len, -1, 3).cpu().numpy()

        if s == 0 or fixed_frame == 0:
            #TODO
            points_all.append(points_gene_np[:, fixed_frame - 1:])
        elif fixed_frame > 0:
            points_all.append(points_gene_np[:, fixed_frame:])

        # fixed_frame = 0 if s == max_step - 1 else sampler_list[s + 1].fixed_frame
        fixed_frame = sampler_list[s].fixed_frame if s == max_step - 1 else sampler_list[s + 1].fixed_frame

        pelvis_new = points_gene[:, -fixed_frame, :9].cpu().numpy().reshape(cfg.batch_size, 3, 3)
        trans_mats = np.repeat(np.eye(4)[np.newaxis, :, :], cfg.batch_size, axis=0)
        for ip, pn in enumerate(pelvis_new):
            _, ret_R, ret_t = rigid_transform_3D(np.matrix(pn), rest_pelvis, False)
            ret_t[1] = 0.0
            rot_euler = R.from_matrix(ret_R).as_euler('zxy')
            shift_euler = np.array([0, 0, rot_euler[2]])
            shift_rot_matrix2 = R.from_euler('zxy', shift_euler).as_matrix()
            trans_mats[ip, :3, :3] = shift_rot_matrix2
            trans_mats[ip, :3, 3] = ret_t.reshape(-1)
        mat = torch.from_numpy(trans_mats).to(device=cfg.device, dtype=torch.float32)

        if fixed_frame > 0:
            fixed_points = points_gene[:, -fixed_frame:]
        if s == max_step - 1:
            print('Saved Mat and Fixed Points', flush=True)
            # np.save(os.path.join(cfg.exp_dir, f'{cfg.method_name}_mat.npy'), mat.cpu().numpy())
            # np.save(os.path.join(cfg.exp_dir, f'{cfg.method_name}_fixed_points.npy'), fixed_points.cpu().numpy())

    points_all = np.concatenate(points_all, axis=1)
    points_all = points_all[:, cnt_seq_len - cnt_fixed_frame:]

    return points_all


def sample_wrapper(trajectory, obj_locs):
    trajectory = convert_trajectory(trajectory)
    # obj_locs = {key: [data[key]['x'], data['key']['z']] for key in data.keys() if 'trajectory' not in key}

    # cfg = compose(config_name="config_sample_synhsi")
    with open('config/config_sample_synhsi.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg = dotDict(cfg)


    # @hydra.main(version_base=None, config_path="../config", config_name="config_sample_synhsi")
    # def sample(cfg) -> None:
    print(cfg)

    # seed_everything(100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_joints_to_smplx = init_model(cfg.model.model_smplx, device=device, eval=True)
    model_joints_to_smplx = JointsToSMPLX(**cfg.model.model_smplx)
    model_joints_to_smplx.load_state_dict(torch.load(cfg.model.model_smplx.ckpt))
    model_joints_to_smplx.to(device)
    model_joints_to_smplx.eval()

    # model_body = init_model(cfg.model.synhsi_body, device=device, eval=True)
    model_body = Unet(**cfg.model.synhsi_body)
    model_body.load_state_dict(torch.load(cfg.model.synhsi_body.ckpt))
    model_body.to(device)
    model_body.eval()
    # model_hand = init_model(cfg.model.synhsi_hand, device=device, eval=True)

    # synhsi_dataset = hydra.utils.instantiate(cfg.dataset)
    synhsi_dataset = TrumansDataset(**cfg.dataset)

    sampler_body = hydra.utils.instantiate(cfg.sampler.pelvis)
    # sampler_hand = hydra.utils.instantiate(cfg.sampler.right_hand)
    sampler_body.set_dataset_and_model(synhsi_dataset, model_body)
    # sampler_hand.set_dataset_and_model(None, model_hand)

    samplers = {'body': sampler_body, 'hand': None}

    # for scene_name in ['N3OpenArea']:

    # trajectory = np.load(os.path.join(cfg.test_dir, cfg.exp_name, f'trajectories.npy'), allow_pickle=True).item()
    # cfg.scene_name = scene_name
    # cfg.action_type = trajectory['action_type']
    # if 'action_id' in trajectory.keys():
    #     cfg.action_id = trajectory['action_id']

    # GP_LAYERS = ['GP_Layer']
    method_name = cfg.method_name
    lid = 0

    base_speed = get_base_speed(cfg, trajectory, is_zup=False)

    mat, goal_list, action_label_list, sampler_list = get_guidance(cfg, trajectory, samplers, act_type=cfg.action_type,
                                                                   speed=int(0.6 * base_speed))
    points_all = sample_step(cfg, mat, obj_locs, goal_list, action_label_list, sampler_list)

    # os.makedirs(cfg.exp_dir, exist_ok=True)
    vertices = None
    for i in range(cfg.batch_size):
        keypoint_gene_torch = torch.from_numpy(points_all[i]).reshape(-1, cfg.dataset.nb_joints * 3).to(device)
        pose, transl, left_hand, right_hand, vertices = joints_to_smpl(model_joints_to_smplx, keypoint_gene_torch, cfg.dataset.joints_ind, cfg.interp_s)
        # output_data = {'transl': transl, 'body_pose': pose[:, 3:], 'global_orient': pose[:, :3],
        #                'id': 0}
        # print(output_data)
        # with open(os.path.join(cfg.exp_dir, f'{method_name}_{lid}_{i}.pkl'), 'wb') as f:
        #     pkl.dump(output_data, f)

    # vertices = np.load('/home/jiangnan/SyntheticHSI/Gradio_demo/vertices.npy', allow_pickle=True)
    # np.save('/home/jiangnan/SyntheticHSI/Gradio_demo/vertices.npy', vertices)

    return vertices.tolist()



    # v = sample()
    #
    #
    # return v




# def load_dataset_meta(cfg):
#     metas = np.load(cfg.)

# if __name__ == '__main__':
    # OmegaConf.register_resolver("times_three", times_three)
    # OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    # sample()
