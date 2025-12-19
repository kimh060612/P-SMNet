# Fine-Structured Learning for Semantic Map Representation

Code for the paper:

**[Fine-Structured Learning for Semantic Map Representation]**
*GiHyeon Kim, DongHyeong Kim, HyeonJeong Lim, MyeongAh Cho*

## Install
The code is tested with Ubuntu 16.04, Python 3.6, Pytorch v1.4+.
 * Install the requirements using pip:

   pip install -r requirements.txt

 * To render egocentric frames in the Matterport3D dataset we use the Habitat simulator. Install [Habitat-sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-lab](https://github.com/facebookresearch/habitat-lab):
 Tested with the following versions Habitat-sim == 0.1.7 and Habitat-lab == 0.1.6.


## Demo
run the following script for demo:

    python demo.py


## Data
 * ```data/paths.json``` has all the manually recorded trajectories.
 * The semantic dense point cloud of objects with cleaned floor labels are available here: https://drive.google.com/drive/folders/1Fwbq7Bvl4kIjJ-YOJNbYWHD_6Gh8lFwQ?usp=sharing. Place those under ```data/object_point_clouds/```. If you are looking to recompute those point clouds you can run ```data/build_point_cloud_from_mesh.py``` (will output a .ply file) or ```data/build_point_cloud_from_mesh_h5.py``` (will output a .h5 file / useful to compute the GT topdown map).
 * Ground truth top-down semantic maps are available here: https://drive.google.com/drive/folders/1aM9vfDckY6K81mrVhVLmEX5rKZ2B1Q5r?usp=sharing. Place those under ```data/semmap/```
 * Place the [Matterport3D](https://niessner.github.io/Matterport/) data under ```data/mp3d/```

## Workflow
 * To recompute the GT topdown semantic maps from the object point clouds (```data/object_point_clouds/```) you can run the following:

        python compute_GT_topdown_semantic_maps/build_semmap_from_obj_point_cloud.py


 * Build training data: (1) build egocentric features + indices, (2) build topdown crops (250x250) (3) preprocess projection indices

        python precompute_training_inputs/build_data.py
        python precompute_training_inputs/build_crops.py
        python precompute_training_inputs/build_projindices.py


 * To train P-SMNet you can run ```train.py```
 * Precompute testing features and projections indices for the full tours in the test set:


        python precompute_test_inputs/build_test_data.py
        python precompute_test_inputs/build_test_data_features.py


 * To evaluate P-SMNet you can run ```test.py``` and:

        python eval/eval.py
        python eval/eval_bfscore.py

## Citation

If you find our work useful in your research, please consider citing:

  @inproceedings{kim2024fine,
    title={Fine-Structured Learning for Semantic Map Representation},
    author={Kim, GiHyeon and Kim, Donghyeong and Lim, Hyeonjeong and Cho, MyeongAh},
    booktitle={2024 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia)},
    pages={1--3},
    year={2024},
    organization={IEEE}
  }

