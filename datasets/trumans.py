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

        # self.body_pose = np.load(os.path.join(folder, 'human_pose.npy'))
        # self.transl = np.load(os.path.join(folder, 'human_transl.npy'))
        # self.global_orient = np.load(os.path.join(folder, 'human_orient.npy'))
        # self.motion_ind = np.load(os.path.join(folder, 'idx_start.npy'))
        # self.joints = np.load(os.path.join(folder, 'human_joints.npy'))
        # self.file_blend = np.load(os.path.join(folder, 'file_blend.npy'))

        self.seq_len=seq_len
        self.step = step
        self.batch_size = batch_size

        # if self.load_action:
        #     self.action_label = np.load(os.path.join(folder, 'action_label.npy')).astype(np.float32)

        if self.load_scene:
            self.mesh_grid = mesh_grid
            self.nb_voxels = nb_voxels
            self.no_objects = no_objects
            self.nb_voxels = nb_voxels
            self.scene_occ = []
            self.scene_dict = {}

            self.scene_folder = os.path.join(folder, 'Scene')
            # self.scene_flag = np.load(os.path.join(folder, 'scene_flag.npy'))
            if not no_objects:
                # self.object_flag = np.load(os.path.join(folder, 'object_flag.npy'))
                # self.object_mat = np.load(os.path.join(folder, 'object_mat.npy'))
                self.object_occ = {}
                self.object_folder = os.path.join(folder, 'Object')
                for file in sorted(os.listdir(self.object_folder)):
                    print(f"Loading object occupied coordinates {file}")
                    obj_name = file.replace('.npy', '')
                    self.object_occ[obj_name] = torch.from_numpy(np.load(os.path.join(self.object_folder, file))).to(device)

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


    def add_object_points(self, points, occ):
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        # voxel = rearrange(voxel, 'b p c -> (b p) c')
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        # voxel = torch.cat([self.batch_id_obj, voxel], dim=-1)
        voxel = voxel[in_bound]
        occ[0, voxel[:, 0], voxel[:, 1], voxel[:, 2]] = True

    def get_occ_for_points(self, points, obj_locs, scene_flag):

        #TODO

        # points_new = points.reshape(-1, 3)
        # center_xz = points_new[:, [0, 2]].mean(axis=0)
        # if torch.norm(center_xz) > 0.:
        #     occ_for_points = torch.load('occ_for_points_at_clear_space.pt').to(points.device)
        #     return occ_for_points


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

        #TODO

        # occ[:] = False
        # occ[:, :, 0, :] = True

        # import cv2
        # img = occ[0, :, 10, :].detach().cpu().numpy()
        # im = np.zeros((300, 400))
        # im[img] = 255
        # cv2.imwrite('gray.jpg', im.T)
        if obj_locs:
            for obj_name, obj_loc in obj_locs.items():
                obj_points = self.object_occ[obj_name].clone()
                obj_points[:, 0] += obj_loc['x']
                obj_points[:, 2] += obj_loc['z']
                # import pdb
                # pdb.set_trace()
                self.add_object_points(obj_points, occ)
        occ_for_points = occ[voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]]
        occ_for_points[torch.logical_not(in_bound)] = True
        occ_for_points = occ_for_points.reshape(batch_size, seq_len, -1)

        # torch.save(occ_for_points, 'occ_for_points_at_clear_space.pt')

        # occ_for_points = torch.ones(batch_size, seq_len, 22).to('cuda')


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

        # aug_z = 0.75 + torch.rand(batch_size, 1) * 0.35
        # grid[:, :, 2] = grid[:, :, 2] * aug_z

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
        import pdb
        data = (data + 1.) * (self.max_torch - self.min_torch) / 2. + self.min_torch
        data = data.reshape(shape_orig)

        return data