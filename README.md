# Scaling Up Dynamic Human-Scene Interaction Modeling

#### This is the code repository of **Scaling Up Dynamic Human-Scene Interaction Modeling** at **CVPR24 (highlight)** 
#### [arXiv](https://arxiv.org/abs/2403.08629) | [Project Page](https://jnnan.github.io/trumans/) | [Dataset](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link) | [Demo](https://huggingface.co/spaces/jnnan/trumans)

# News
- RGB videos in 6 FPS and smplx parameters in camera coordinate can be [download](https://drive.google.com/drive/folders/1axXjugOzxlWD_kJsY6v-cn5Q7k5V1YfL?usp=sharing)
- All the interactable objects in TRUMANS dataset are released.

# Human Motion Synthesis in Editable Indoor Scenes

This project provides an implementation of our method for synthesizing human motion in an indoor scene, based on a user-defined trajectory. The furniture configuration within the scene can be edited freely by the user. This is achieved by using a [Flask](https://flask.palletsprojects.com/en/3.0.x/quickstart/) application to facilitate the interactive input and visualization.

## Features

- **Editable Indoor Scene**: Users can modify the furniture configuration within the indoor scene.
- **Trajectory Drawing**: Users can draw a trajectory within the scene.
- **Human Motion Synthesis**: The application generates human motion based on the drawn trajectory.

## Getting Started

### Prerequisites

To run the application, you need to have the following installed:

- Python 3.x
- Flask
- Required Python packages (specified in `requirements.txt`)

### Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/jnnan/trumans_utils.git
    cd trumans_utils
    ```

2. **Download Checkpoints, Data, and SMPL-X Models**:
    - Download the necessary files and folders from [this link](https://drive.google.com/file/d/1sjfaUTg2pv7VEjQYwk313vZSbG_kXPv0/view?usp=sharing).
    - Extract trumans_demo.zip, and place the four folders at the root of the project directory (./trumans_utils).

3. **Install Python Packages**:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

To start the Flask application:

```sh
python3 -m flask run --host=0.0.0.0
```

The application will be available at `http://127.0.0.1:5000`.

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000`.
2. You will see an interface where you can edit the indoor scene configuration.
3. Draw a trajectory within the scene.
4. The application will synthesize human motion based on the drawn trajectory and display it within the scene.


# TRUMANS Dataset

Please download the TRUMANS dataset from [Google Drive](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link). The content inside the download link will be continuously updated to ensure you have access to the most recent data.

Explanation of the files and folders of the TRUMANS dataset:

- **Scene (folder):** This folder contains the occupancy grids for the scenes in the TRUMANS dataset. The coverage range is from -3 to 3 meters along the x-axis, 0 to 2 meters along the y-axis, and -4 to 4 meters along the z-axis. The voxel shape is (300, 100, 400) in the x, y, and z axis order, using a y-up coordinate system.
- **Scene_mesh (folder):** This folder includes the OBJ mesh files of the scenes.
- **Scene_occ_render (folder):** This folder contains the top-down view renders of the scene occupancy grids.
- **Object_chairs:** 
    - **Object (folder):** This folder holds the occupancy grids for the sittable objects in the TRUMANS dataset. The format is the same as that of the scenes.
    - **Object_mesh (folder):** This folder contains the OBJ mesh files of the objects.
- **Object_all:** All interactable objects.
    - **Object_pose:** Object rotation (xyz Euler) and location (y-up coordinate system) for each motion segment.
    - **Object_mesh:** OBJ mesh of the objects.
- **human_pose.npy:** This file contains a (3,792,068 x 63) array, where each row corresponds to the 63-dimensional SMPL-X body_pose parameter of one frame of MoCap data. The data is a concatenation of all motion segments.
- **human_orient.npy:** This file contains a (3,792,068 x 3) array corresponding to the global_orient parameter of SMPL-X.
- **human_transl.npy:** This file contains a (3,792,068 x 3) array corresponding to the transl parameter of SMPL-X.
- **seg_name.npy:** This file contains a (3,792,068,) string array, where each string represents the motion segment name of a particular frame.
- **frame_id.npy:** This file contains a (3,792,068,) integer array, with each value representing the frame number of the original motion segment to which the current index belongs.
- **scene_list.npy:** This is a list of scene names corresponding to the files in the "Scene" folder.
- **scene_flag.npy:** This file contains a (3,792,068,) integer array. Each value indicates the scene name of the current frame by indexing the  `scene_list.npy`.
- **object_list.npy:** This is a list of object names corresponding to the files in the "Object" folder.
- **object_flag.npy:** This file contains a (3,792,068 x 32) integer array. Each row has 32 values, indicating the presence of 32 objects for each frame index. A value of -1 means the corresponding object does not apply, while a value equal to or greater than 1 indicates the presence of the object in that frame. The value is linked to the  `object_mat.npy`  file.
- **object_mat.npy:** This file contains a (3,792,068 x 5 x 4 x 4) array. The second dimension (with shape 5) corresponds to the 5 types of objects, whose existence is indicated in the  `object_flag.npy`  file. The last two dimensions represent the rotation and translation of the object in a 4x4 matrix.
- **bad_frames.npy:** This is a list of frames that contain erroneous MoCap data.

#### Note: The data associated with action labels and 2D rendering will be uploaded soon.

# Citation
```
@article{jiang2024scaling,
  title={Scaling up dynamic human-scene interaction modeling},
  author={Jiang, Nan and Zhang, Zhiyuan and Li, Hongjie and Ma, Xiaoxuan and Wang, Zan and Chen, Yixin and Liu, Tengyu and Zhu, Yixin and Huang, Siyuan},
  journal={arXiv preprint arXiv:2403.08629},
  year={2024}
}
```
