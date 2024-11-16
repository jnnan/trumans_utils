# Scaling Up Dynamic Human-Scene Interaction Modeling

#### This is the code repository of **Scaling Up Dynamic Human-Scene Interaction Modeling** at **CVPR24 (highlight)** 
#### [arXiv](https://arxiv.org/abs/2403.08629) | [Project Page](https://jnnan.github.io/trumans/) | [Dataset](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link) | [Demo](https://huggingface.co/spaces/jnnan/trumans)

# News
- 🎉🎉🎉 **The meshes and textures are released!** 🎉🎉🎉 Download the full dataset from [Google Drive](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link) and the scenes are in the **Recoerdings_blend** folder. Each .blend file contains 10 motion-captured sequences in that scene.

  How to Use:
  1. Download blender from the [official page](https://www.blender.org/download/)
  2. Download our [addon](https://github.com/jnnan/trumans_utils/blob/main/HSI_addon-zzy.zip) for data visualization
  3. Install the addon, and navigate the scenes following our tutorial:

https://github.com/user-attachments/assets/3a510469-0146-4e14-a259-54366168001a

- [Training code](#training) has been updated
- **The Action Annotation is released!** Download the full dataset from [Google Drive](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link) and the action annotation is in the **Actions** folder, with start frame, end frame and textual description.
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

# Training
## Overview

This README provides instructions on setting up and training the TRUMANS model using the provided dataset.

## Prerequisites

Before you begin, make sure you have the following software installed:
```sh
    pip install -r requirements.txt
```

## Dataset Setup

1. Download the TRUMANS dataset from the provided link.
2. Place the dataset files in the following directory within your project:
   ```
   ./trumans/Data_release
   ```

## Configuration

Set the `ROOT_DIR` environment variable to the absolute path of the `./trumans` directory in your system. This can be done by adding the following line to your `.bashrc` or `.bash_profile`:

```bash
export ROOT_DIR='/absolute/path/to/trumans'
```

Make sure to replace `/absolute/path/to/trumans` with the actual path to the `trumans` folder on your machine.

## Model Training

Navigate to the `trumans` directory:

```bash
cd trumans
```

To start training the model, run the training script from the command line:

```bash
python train_synhsi.py
```

The training script will automatically load the dataset from `Data_release`, set up the model, and commence training sessions using the configurations in ./trumans/config folder.

## Annotation

The dataset includes an `action_label.npy` file containing frame-wise annotations for the motions. The labels correspond to the type of interaction and are indexed as follows:

| Interaction Type | Label |
|------------------|-------|
| Lie down         | 0     |
| Squat            | 1     |
| Mouse            | 2     |
| Keyboard         | 3     |
| Laptop           | 4     |
| Phone            | 5     |
| Book             | 6     |
| Bottle           | 7     |
| Pen              | 8     |
| Vase             | 9     |

These labels are used during the training to provide supervision for the learning model. To visualize the generated motion according to the predefined labels, see the [demo part](#human-motion-synthesis-in-editable-indoor-scenes) and change line 81 of [sample_hsi.py](https://github.com/jnnan/trumans_utils/blob/main/sample_hsi.py)


# TRUMANS Dataset

Please download the TRUMANS dataset from [Google Drive](https://docs.google.com/forms/d/e/1FAIpQLSdF62BQ9TQTSTW0HzyNeRPhlzmREL5T8hUGn-484W1I3eVihQ/viewform?usp=sf_link). The content inside the download link will be continuously updated to ensure you have access to the most recent data.

Explanation of the files and folders of the TRUMANS dataset:

- **Action (folder):** This folder contains the action annotation in language for each motion segment, indicated by each file name.
- **smplx_result (folder):** SMPL-X annotation for each motion segment in global coordinate.
- **smplx_result_in_cam (folder):** SMPL-X annotation for each motion segment in the camera coordinate of corresponding frames.
- **video_render (folder):** RGB video rendering of each motion segment in 30 FPS.
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
- **left_hand_pose.npy:** Left hand pose in SMPL-X.
- **right_hand_pose.npy:** Right hand pose in SMPL-X.
- **seg_name.npy:** This file contains a (3,792,068,) string array, where each string represents the motion segment name of a particular frame.
- **frame_id.npy:** This file contains a (3,792,068,) integer array, with each value representing the frame number of the original motion segment to which the current index belongs.
- **scene_list.npy:** This is a list of scene names corresponding to the files in the "Scene" folder.
- **scene_flag.npy:** This file contains a (3,792,068,) integer array. Each value indicates the scene name of the current frame by indexing the  `scene_list.npy`.
- **object_list.npy:** This is a list of object names corresponding to the files in the "Object" folder.
- **object_flag.npy:** This file contains a (3,792,068 x 35) integer array. Each row has 35 values, indicating the presence of 35 objects for each frame index. A value of -1 means the corresponding object does not apply, while a value equal to or greater than 1 indicates the presence of the object in that frame. The value is linked to the  `object_mat.npy`  file.
- **object_mat.npy:** This file contains a (3,792,068 x 5 x 4 x 4) array. The second dimension (with shape 5) corresponds to the 5 types of objects, whose existence is indicated in the  `object_flag.npy`  file. The last two dimensions represent the rotation and translation of the object in a 4x4 matrix.
- **bad_frames.npy:** This is a list of frames that contain erroneous MoCap data.

#### Note: The data associated with action labels and 2D rendering will be uploaded soon.

# Citation
```
@inproceedings{jiang2024scaling,
  title={Scaling up dynamic human-scene interaction modeling},
  author={Jiang, Nan and Zhang, Zhiyuan and Li, Hongjie and Ma, Xiaoxuan and Wang, Zan and Chen, Yixin and Liu, Tengyu and Zhu, Yixin and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1737--1747},
  year={2024}
}
```
