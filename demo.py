import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

from SMNet.model_test import SMNet

from projector import _transform3D
from projector.projector import Projector
from scipy.spatial.transform import Rotation as R

from utils.habitat_utils import HabitatUtils
from utils import convert_weights_cuda_cpu

from semseg.rednet import RedNet
import cv2


# envs = [
#     'JF19kD82Mey_1', 'wc2JMjhGNzB_1', 'i5noydFURQK_0', 'cV4RVeZvu5T_0', '8194nk5LbLH_0', 'rqfALeAoiTq_2', '759xd9YjKW5_1', 'VVfe2KiqLaN_1', 'JF19kD82Mey_0', 'GdvgFV5R1Z5_0', 'ur6pFq6Qu1A_0', 'ac26ZMwG7aT_0', '1LXtFkjw3qL_0', 'EDJbREhghzL_0', 'jh4fc5c5qoQ_0', 'uNb9QFRL6hY_0', '5LpN3gDmAk7_0', 'zsNo4HB9uLZ_0', 'YFuZgdQ5vWj_0', 'E9uDoFAP3SH_0', 'p5wJjkQkbXX_1', 'X7HyMhZNoso_0', 'VVfe2KiqLaN_0', '1pXnuDYAj8r_1', '5ZKStnWn8Zo_1', 'HxpKQynjfin_0', '5q7pvUzZiYa_1', 'q9vSo1VnCiC_0', 'rPc6DW4iMge_0', 'r47D5H71a5s_0', '1LXtFkjw3qL_1', 'Z6MFQCViBuw_0', 'x8F5xyUWy9e_0', '29hnd4uzFmX_0', 'Vvot9Ly1tCj_0', 'fzynW3qQPVF_1', 'aayBHfsNo7d_0', '29hnd4uzFmX_1', 'ZMojNkEp431_0', 'QUCTc6BB5sX_0', 'B6ByNegPMKs_0', 'YVUC4YcDtcY_0', 'S9hNv5qa7GM_0', '17DRP5sb8fy_0', 'mJXqzFtmKg4_0', '2t7WUuJeko7_0', '8WUmhLawc2A_0', 'ULsKaCPVFJR_0', 'PuKPg4mmafe_0', 'wc2JMjhGNzB_0', 'S9hNv5qa7GM_1', 'Vt2qJdWjCF2_0', 'VLzqgDo317F_1', 'sT4fr6TAbpF_0', '5q7pvUzZiYa_2', 'RPmz2sHmrrY_0', 'YFuZgdQ5vWj_1', 'QUCTc6BB5sX_1', 'ARNzJeq3xxb_0', 'Uxmj2M2itWa_0', 'aayBHfsNo7d_1', 'rPc6DW4iMge_1', '2n8kARJN3HM_1', 'ZMojNkEp431_1', '2n8kARJN3HM_0', '759xd9YjKW5_0', '82sE5b5pLXE_0', 'i5noydFURQK_1', '5ZKStnWn8Zo_0', 'ULsKaCPVFJR_1', 'cV4RVeZvu5T_1', 'rqfALeAoiTq_1', 'fzynW3qQPVF_0', 'UwV83HsGsw3_1', 'e9zR4mvMWw7_1', 'jtcxE69GiFV_1', 'Pm6F8kyY3z2_1', 'JeFG25nYj2p_0', '1pXnuDYAj8r_0', 'jtcxE69GiFV_0', 'UwV83HsGsw3_0', 'r1Q1Z4BcV1o_0', '5q7pvUzZiYa_0', 'JmbYfDe2QKZ_0', 'E9uDoFAP3SH_1', 'JmbYfDe2QKZ_1', 'e9zR4mvMWw7_2', '1LXtFkjw3qL_2', 'WYY7iVyf5p8_1'
# ]
envs = [
    'q9vSo1VnCiC_0'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Settings
resolution = 0.02 # topdown resolution
default_ego_dim = (480, 640) #egocentric resolution
z_clip = 0.50 # detections over z_clip will be ignored
vfov = 67.5
vfov = vfov * np.pi / 180.0

# -- load JSONS 
info = json.load(open('data/semmap_GT_info.json','r'))
paths = json.load(open('data/paths.json', 'r'))

# -- Create RedNet model
cfg_rednet = {
    'arch': 'rednet',
    'resnet_pretrained': False,
    'finetune': True,
    'SUNRGBD_pretrained_weights': '',
    'n_classes': 13,
    'upsample_prediction': True,
    'load_model': 'rednet_mp3d_best_model.pkl',
}

model_rednet = RedNet(cfg_rednet)
model_rednet = model_rednet.to(device)

print('Loading pre-trained weights: ', cfg_rednet['load_model'])
state = torch.load(cfg_rednet['load_model'])
model_state = state['model_state']
model_state = convert_weights_cuda_cpu(model_state, 'cpu')
model_rednet.load_state_dict(model_state)
model_rednet.eval()

normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])
# -- create SMNet model
cfg_model = {
    'arch': 'smnet',
    'finetune': False,
    'n_obj_classes': 13,
    'ego_feature_dim': 64,
    'mem_feature_dim': 256,
    'mem_update': 'gru',
    'ego_downsample': False,
}
model_path = 'smnet_mp3d_best_model.pkl'

model = SMNet(cfg_model, device)
model = model.to(device)

print('Loading pre-trained weights: ', model_path)
state = torch.load(model_path)
model_state = state['model_state']
model_state = convert_weights_cuda_cpu(model_state, 'cpu')
model.load_state_dict(model_state)
model.eval()


for env in tqdm(envs):
    # -- instantiate Habitat
    house, level = env.split('_')
    scene = 'data/mp3d/{}/{}.glb'.format(house, house)
    habitat = HabitatUtils(scene, int(level))

    # -- get house info
    world_dim_discret = info[env]['dim']
    map_world_shift = info[env]['map_world_shift']
    map_world_shift = np.array(map_world_shift)
    world_shift_origin=torch.from_numpy(map_world_shift).float().to(device=device)

    # -- instantiate projector
    projector = Projector(vfov, 1,
                        default_ego_dim[0],
                        default_ego_dim[1],
                        world_dim_discret[2], # height
                        world_dim_discret[0], # width
                        resolution,
                        world_shift_origin,
                        z_clip,
                        device=device)

    # compute projections indices and egocentric features
    path = paths[env]

    N = len(path['positions'])

    projections_wtm = np.zeros((N,480,640,2), dtype=np.uint16)
    projections_masks = np.zeros((N,480,640), dtype=np.bool)
    projections_heights = np.zeros((N,480,640), dtype=np.float32)

    features_lastlayer = np.zeros((N,64,240,320), dtype=np.float32)

    print('Compute egocentric features and projection indices')

    with torch.no_grad():
        for n in tqdm(range(N)):
            pos = path['positions'][n]
            ori = path['orientations'][n]

            habitat.position = list(pos)
            habitat.rotation = list(ori)
            habitat.set_agent_state()

            sensor_pos = habitat.get_sensor_pos()
            sensor_ori = habitat.get_sensor_ori()

            # -- get T transorm
            sensor_ori = np.array([sensor_ori.x, sensor_ori.y, sensor_ori.z, sensor_ori.w])
            r = R.from_quat(sensor_ori)
            elevation, heading, bank = r.as_rotvec()

            xyzhe = np.array([[sensor_pos[0],
                            sensor_pos[1],
                            sensor_pos[2],
                            heading,
                            elevation + np.pi]])

            xyzhe = torch.FloatTensor(xyzhe).to(device)
            T = _transform3D(xyzhe, device=device)

            # -- depth for projection
            depth = habitat.render(mode='depth')
            depth = depth[:,:,0]
            depth = depth.astype(np.float32)
            depth *= 10.0
            depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(device)

            # -- projection
            world_to_map, mask_outliers, heights = projector.forward(depth_var, T, return_heights=True)

            world_to_map = world_to_map[0].cpu().numpy()
            mask_outliers = mask_outliers[0].cpu().numpy()
            heights = heights[0].cpu().numpy()

            world_to_map = world_to_map.astype(np.uint16)
            mask_outliers = mask_outliers.astype(np.bool)
            heights = heights.astype(np.float32)

            projections_wtm[n,...] = world_to_map
            projections_masks[n,...] = mask_outliers
            projections_heights[n,...] = heights
    
            # -- get egocentric features
            rgb = habitat.render()
            rgb = rgb.astype(np.float32)
            rgb = rgb / 255.0
            rgb = torch.FloatTensor(rgb).permute(2,0,1)
            rgb = normalize(rgb)
            rgb = rgb.unsqueeze(0).to(device)

            depth_enc = habitat.render(mode='depth')
            depth_enc = depth_enc[:,:,0]
            depth_enc = depth_enc.astype(np.float32)
            depth_enc = torch.FloatTensor(depth_enc).unsqueeze(0)
            depth_enc = depth_normalize(depth_enc)
            depth_enc = depth_enc.unsqueeze(0).to(device)

            semfeat_lastlayer = model_rednet(rgb, depth_enc)
            semfeat_lastlayer = semfeat_lastlayer[0].cpu().numpy()
            semfeat_lastlayer = semfeat_lastlayer.astype(np.float32)
            features_lastlayer[n,...] = semfeat_lastlayer

    del habitat, projector

    print('Run SMNet')

    with torch.no_grad():
        
        # get env dim
        world_dim_discret = info[env]['dim']
        map_height = world_dim_discret[2]
        map_width  = world_dim_discret[0]
        
        mask_outliers = projections_masks
        heights = projections_heights
        features = features_lastlayer
        
        features = torch.from_numpy(features)
        
        projections_wtm = projections_wtm.astype(np.int32)
        projections_wtm = torch.from_numpy(projections_wtm)
        mask_outliers = torch.from_numpy(mask_outliers)
        heights = torch.from_numpy(heights)
        
        scores, observed_map, height_map = model(features,
                                                projections_wtm,
                                                mask_outliers,
                                                heights,
                                                map_height,
                                                map_width)
        
        semmap = scores.data.max(0)[1]
        semmap[~observed_map] = 0
        semmap = semmap.cpu().numpy()
        semmap = semmap.astype(np.uint8)
        scores = scores.cpu().numpy()
        observed_map = observed_map.cpu().numpy()
        height_map = height_map.cpu().numpy()

        from utils.semantic_utils import color_label
        semmap_color = color_label(semmap)
        semmap_color = semmap_color.transpose(1,2,0)
        semmap_color = semmap_color.astype(np.uint8)

        cv2.imwrite("map_viz/output_smnet{}.png".format(env), semmap_color)    
        # import matplotlib.pyplot as plt 
        # plt.imshow(semmap_color)
        # plt.title('Topdown semantic map prediction')
        # plt.axis('off')
        # plt.show()
