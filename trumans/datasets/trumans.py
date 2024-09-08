import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader, Subset


class TrumansDataset(Dataset):
    def __init__(self, folder, device, mesh_grid, batch_size=1, seq_len=32, step=1, nb_voxels=32, train=True, load_scene=True, load_action=True, no_objects=False, **kwargs):
        self.device = device
        self.train = train
        self.load_scene = load_scene
        self.load_action = load_action

        self.global_orient = np.load(os.path.join(folder, 'human_orient.npy'))
        self.motion_ind = np.load(os.path.join(folder, 'frame_id.npy'))
        self.joints = np.load(os.path.join(folder, 'human_joints.npy'))

        self.seq_len=seq_len
        self.step = step
        self.batch_size = batch_size

        if self.load_action:
            self.action_label = np.load(os.path.join(folder, 'action_label.npy')).astype(np.float32)

        if self.load_scene:
            self.mesh_grid = mesh_grid
            self.nb_voxels = nb_voxels
            self.no_objects = no_objects
            self.nb_voxels = nb_voxels
            self.scene_occ = []
            self.scene_dict = {}
            if train:
                self.scene_folder = os.path.join(folder, 'Scene')
                self.scene_flag = np.load(os.path.join(folder, 'scene_flag.npy'))
                if not no_objects:
                    self.object_flag = np.load(os.path.join(folder, 'object_flag.npy'))
                    self.object_mat = np.load(os.path.join(folder, 'object_mat.npy'))
                    self.object_occ = []
                    self.object_folder = os.path.join(folder, 'Object_chairs', 'Object')
                    for file in sorted(os.listdir(self.object_folder)):
                        print(f"Loading object occupied coordinates {file}")
                        self.object_occ.append(np.load(os.path.join(self.object_folder, file)))
            else:
                self.scene_folder = os.path.join(folder, 'Scene_test')
            for sid, file in enumerate(sorted(os.listdir(self.scene_folder))):
                # if scene_name != '' and scene_name not in file:
                #     continue
                print(f"{sid} Loading Scene Mesh {file}")
                scene_occ = np.load(os.path.join(self.scene_folder, file))
                scene_occ = torch.from_numpy(scene_occ).to(device=device, dtype=bool)
                self.scene_occ.append(scene_occ)
                self.scene_dict[file] = sid
            self.scene_occ = torch.stack(self.scene_occ)

            self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
            self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(device)
            self.batch_id = torch.linspace(0, batch_size - 1, batch_size).tile((nb_voxels ** 3, 1)).T\
                .reshape(-1, 1).to(device=device, dtype=torch.long)
            self.batch_id_obj = torch.linspace(0, batch_size - 1, batch_size).tile((9000, 1)).T \
                .reshape(-1, 1).to(device=device, dtype=torch.long)

         # TODO CHANGE STEP
        norm = np.load(os.path.join(folder, 'norm.npy'), allow_pickle=True).item()[f'{seq_len, 3}']
        self.min = norm[0].astype(np.float32)
        self.max = norm[1].astype(np.float32)
        self.min_torch = torch.tensor(self.min).to(device)
        self.max_torch = torch.tensor(self.max).to(device)

    def __getitem__(self, idx):
        idx = self.motion_ind[idx]
        joints = self.joints[idx: idx + self.step * self.seq_len: self.step]
        # joints_orig = joints.copy()
        init_joints = np.array([joints[0, 0, 0], 0., joints[0, 0, 2]]) # joints[0][0] #joints[0][0]
        joints = joints - init_joints
        global_orient = self.global_orient[idx: idx + self.step * self.seq_len: self.step]
        if self.load_action:
            action_label = self.action_label[idx: idx + self.step * self.seq_len: self.step]
        else:
            action_label = 0
        # body_pose = self.body_pose[idx: idx + self.step * self.seq_len: self.step]

        init_global_orient = global_orient[0]
        init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')
        shift_euler = np.array([0, 0, -init_global_orient_euler[2]])
        shift_rot_matrix = R.from_euler('zxy', shift_euler).as_matrix()

        mat = np.eye(4)
        mat[:3, :3] = np.linalg.inv(shift_rot_matrix.T).T
        mat[:3, 3] = init_joints
        mat = mat.astype(np.float32)

        joints = joints @ shift_rot_matrix.T
        joints = self.normalize(joints)
        joints = joints.astype(np.float32).reshape((self.seq_len, -1))

        if self.train and self.load_scene:
            scene_flag = self.scene_flag[idx]
        else:
            scene_flag = 0

        if self.load_scene and not self.no_objects:
            if self.train and np.any(self.object_flag[idx] >= 0):
                obj_points_all = []
                for obj_id, has in enumerate(self.object_flag[idx]):
                    if has >= 0:
                        obj_points = self.object_occ[obj_id]
                        obj_mat = self.object_mat[idx][has]
                        obj_points = obj_points @ obj_mat[:3, :3] + obj_mat[:3, 3]
                        obj_points_all.append(obj_points)
                obj_points_all = np.concatenate(obj_points_all, axis=0)
                try:
                    obj_points_all = np.pad(obj_points_all, ((0, 9000 - obj_points_all.shape[0]), (0, 0)), 'constant', constant_values=0)
                except:
                    print(obj_points_all.shape[0], flush=True)
                    print(0)
            else:
                obj_points_all = np.zeros((9000, 3))
        else:
            obj_points_all = np.zeros((9000, 3))

        return joints, obj_points_all, mat, scene_flag, action_label

    def add_object_points(self, points, occ):
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        # voxel = rearrange(voxel, 'b p c -> (b p) c')
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel = torch.cat([self.batch_id_obj, voxel], dim=-1)
        voxel = voxel[in_bound]
        occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]] = True

    def get_occ_for_points(self, points, obj_points, scene_flag):

        if isinstance(scene_flag, str):
            for k, v in self.scene_dict.items():
                if scene_flag in k:
                    scene_flag = [v]
                    break
        batch_size = points.shape[0]
        seq_len = points.shape[1]
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        # voxel = rearrange(voxel, 'b p c -> (b p) c')
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0
        voxel = torch.cat([self.batch_id, voxel], dim=1)
        occ = self.scene_occ[scene_flag]

        if obj_points is not None:
            self.add_object_points(obj_points, occ)
        occ_for_points = occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]]
        occ_for_points[torch.logical_not(in_bound)] = True
        occ_for_points = occ_for_points.reshape(batch_size, seq_len, -1)

        return occ_for_points

    def create_meshgrid(self, batch_size=1):
        bbox = self.mesh_grid
        size = (self.nb_voxels, self.nb_voxels, self.nb_voxels)
        x = torch.linspace(bbox[0], bbox[1], size[0])
        y = torch.linspace(bbox[2], bbox[3], size[1])
        z = torch.linspace(bbox[4], bbox[5], size[2])
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        grid = grid.repeat(batch_size, 1, 1)

        return grid


    @staticmethod
    def combine_mesh(vert_list, face_list):
        assert len(vert_list) == len(face_list)
        verts = None
        faces = None
        for v, f in zip(vert_list, face_list):
            if verts is None:
                verts = v
                faces = f
            else:
                f = f + verts.shape[0]
                verts = torch.cat([verts, v])
                faces = torch.cat([faces, f])

        return verts, faces

    @staticmethod
    def transform_mesh(vert_list, trans_mats):
        assert len(vert_list) == len(trans_mats)
        vert_list_new = []
        for v, m in zip(vert_list, trans_mats):
            v = v @ m[:3, :3].T + m[:3, 3]
            vert_list_new.append(v)
        vert_list_new = torch.stack(vert_list_new)

        return vert_list_new

    def __len__(self):
        return len(self.motion_ind)


    def normalize(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        # data = (data - self.mean) / self.std
        data = -1. + 2. * (data - self.min) / (self.max - self.min)
        data = data.reshape(shape_orig)

        return data

    def normalize_torch(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        # data = (data - self.mean) / self.std
        data = -1. + 2. * (data - self.min_torch) / (self.max_torch - self.min_torch)
        data = data.reshape(shape_orig)

        return data

    def denormalize(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        # data = data * self.std + self.mean
        data = (data + 1.) * (self.max - self.min) / 2. + self.min
        data = data.reshape(shape_orig)

        return data

    def denormalize_torch(self, data):
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        # data = data * self.std + self.mean
        data = (data + 1.) * (self.max_torch - self.min_torch) / 2. + self.min_torch
        data = data.reshape(shape_orig)

        return data