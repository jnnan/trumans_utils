# Scaling Up Dynamic Human-Scene Interaction Modeling

#### This is the code repository of **Scaling Up Dynamic Human-Scene Interaction Modeling** at **CVPR24 (highlight)** 
#### [arXiv](https://arxiv.org/abs/2403.08629) | [Project Page](https://jnnan.github.io/trumans/) | [Dataset](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link) | [Demo](https://huggingface.co/spaces/jnnan/trumans)

## TRUMANS Dataset

Please download the TRUMANS dataset from [Google Drive](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link). The content inside the download link will be continuously updated to ensure you have access to the most recent data.

Here's the revised explanation of the files and folders of the TRUMANS dataset:

- **Scene (folder):** This folder contains the occupancy grids for the scenes in the TRUMANS dataset. The coverage range is from -3 to 3 meters along the x-axis, 0 to 2 meters along the y-axis, and -4 to 4 meters along the z-axis. The voxel shape is (300, 100, 400) in the x, y, and z axis order, using a y-up coordinate system.
- **Scene (folder):** This folder includes the OBJ mesh files of the scenes.
- **Scene_occ_render (folder):** This folder contains the top-down view renders of the scene occupancy grids.
- **Object (folder):** This folder holds the occupancy grids for the sittable objects in the TRUMANS dataset. The format is the same as that of the scenes.
- **Object_mesh (folder):** This folder contains the OBJ mesh files of the objects.
- **human_pose.npy:** This file contains a (3,792,068 x 63) array, where each row corresponds to the 63-dimensional SMPL-X body_pose parameter of one frame of MoCap data. The data is a concatenation of all motion segments.
- **human_orient.npy:** This file contains a (3,792,068 x 3) array corresponding to the global_orient parameter of SMPL-X.
- **human_transl.npy:** This file contains a (3,792,068 x 3) array corresponding to the transl parameter of SMPL-X.
- **seg_name:** This file contains a (3,792,068,) string array, where each string represents the motion segment name of a particular frame.
- **frame_id:** This file contains a (3,792,068,) integer array, with each value representing the frame number of the original motion segment to which the current index belongs.
- **scene_list:** This is a list of scene names corresponding to the files in the "Scene" folder.
- **scene_flag:** This file contains a (3,792,068,) integer array. Each value indicates the scene name of the current frame by indexing the  `scene_list.npy`.
- **object_list:** This is a list of object names corresponding to the files in the "Object" folder.
- **object_flag:** This file contains a (3,792,068 x 32) integer array. Each row has 32 values, indicating the presence of 32 objects for each frame index. A value of -1 means the corresponding object does not apply, while a value equal to or greater than 1 indicates the presence of the object in that frame. The value is linked to the  `object_mat.npy`  file.
- **object_mat:** This file contains a (3,792,068 x 5 x 4 x 4) array. The second dimension (with shape 5) corresponds to the 5 types of objects, whose existence is indicated in the  `object_flag.npy`  file. The last two dimensions represent the rotation and translation of the object in a 4x4 matrix.
- **bad_frames:** This is a list of frames that contain erroneous MoCap data.

#### Note: The data associated with action labels will be uploaded soon.

## Citation
```
@article{jiang2024scaling,
  title={Scaling up dynamic human-scene interaction modeling},
  author={Jiang, Nan and Zhang, Zhiyuan and Li, Hongjie and Ma, Xiaoxuan and Wang, Zan and Chen, Yixin and Liu, Tengyu and Zhu, Yixin and Huang, Siyuan},
  journal={arXiv preprint arXiv:2403.08629},
  year={2024}
}
```
