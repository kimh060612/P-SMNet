import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_max
from SMNet.probabilistic_net import AxisAlignedGaussian, Fcomb
from SMNet.unet import UNet

class PSMNet(nn.Module):
    def __init__(self, cfg, device):
        super(PSMNet, self).__init__()
        
        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']
        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = device  # cpu
        #self.device_mem = torch.device('cuda')  # cpu

        if mem_update == 'lstm':
            self.rnn = nn.LSTMCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)

        elif mem_update == 'gru':
            self.rnn = nn.GRUCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)
        elif mem_update == 'replace':
            self.linlayer = nn.Linear(ego_feat_dim, mem_feat_dim)
        else:
            raise Exception('{} memory update not supported.'.format(mem_update))

        self.input_channels = mem_feat_dim
        self.num_classes = n_obj_classes
        self.num_filters = [32,64,128,192]
        self.latent_dim = 6
        self.no_convs_per_block = 3
        self.no_convs_fcomb = 4
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = 10.
        self.z_prior_sample = 0
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.prior = AxisAlignedGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers, self.num_classes).to(device)
        self.posterior = AxisAlignedGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, self.num_classes, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)

        self.unet = UNet(self.mem_feat_dim, self.num_classes, self.num_filters, self.initializers , apply_last_layer=False, padding=True).to(device)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    

    def encode(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        T,C,H,W = features.shape

        mask_inliers = ~mask_outliers

        memory_size = map_height * map_width * self.mem_feat_dim * 4 / 1e9
        if memory_size > 5:
            self.device_mem = torch.device('cpu')
        else:
            self.device_mem = torch.device('cuda')
        self.decoder = self.decoder.to(self.device_mem)

        if self.mem_update == 'lstm':
            state = (torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
                     torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((map_height*map_width), dtype=torch.bool, device=self.device)
        height_map = torch.zeros((map_height*map_width), dtype=torch.float, device=self.device)

        for t in tqdm(range(T)):

            feature = features[t,:,:,:]
            world_to_map = proj_wtm[t,:,:,:]
            inliers = mask_inliers[t,:,:]
            height = heights[t,:,:]

            world_to_map = world_to_map.long()

            feature = feature.to(self.device)
            world_to_map = world_to_map.to(self.device)
            inliers = inliers.to(self.device)
            height = height.to(self.device)


            if self.ego_downsample:
                world_to_map = world_to_map[::4, ::4, :]
                inliers = inliers[::4,::4]
                height = height[::4,::4]

            flat_indices = (map_width*world_to_map[:,:,1] + world_to_map[:,:,0]).long()
            flat_indices = flat_indices[inliers]
            height = height[inliers]
            height += 1000
            height_map, highest_height_indices = scatter_max(
                height,
                flat_indices,
                dim=0,
                out = height_map,
            )

            m = highest_height_indices >= 0

            observed_masks += m

            if m.any():
                feature = F.interpolate(feature.unsqueeze(0), size=(480,640), mode="bilinear", align_corners=True)
                feature = feature.squeeze(0)
                if self.ego_downsample:
                    feature = feature[:,::4,::4]

                feature = feature.permute(1,2,0)  # -- (N,H,W,512)

                feature = feature[inliers, :]

                tmp_memory = feature[highest_height_indices[m], :]

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][m,:].to(self.device),
                                 state[1][m,:].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][m,:] = tmp_state[0].to(self.device_mem)
                    state[1][m,:] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[m,:].to(self.device)

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[m,:] = tmp_state.to(self.device_mem)
                else:
                    raise NotImplementedError


                del tmp_memory
            del feature

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            memory = state
        
        memory = memory.view(map_height, map_width, self.mem_feat_dim)

        memory = memory.permute(2,0,1)
        memory = memory.unsqueeze(0)

        return memory, observed_masks, height_map

    def prior_samples(self, map_state):
        sample = self.prior.forward(map_state)
        return sample
    
    # def posterior_sample(self, map_state, gt_map):
    #     gt_map = self.one_hot_encoding(gt_map)
    #     # print(map_state.shape, gt_map.shape)
    #     sample = self.posterior.forward(map_state, gt_map)
    #     return sample
        
    def sample(self, map_memory, latent_space):
        return self.fcomb.forward(map_memory, latent_space.sample())
    
    # def one_hot_encoding(self, gt_map):
    #     return F.one_hot(gt_map, num_classes=self.num_classes).permute(0, 3, 1, 2)
    
    def forward(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        prior_latent_space = self.prior_samples(features)
        memory, observed_masks, height_map = self.encode(
            features, 
            proj_wtm, 
            mask_outliers, 
            heights, 
            map_height, 
            map_width
        )
        prior_latent_space = self.prior_samples(memory)
        semmap = self.unet(memory, False)
        semmap_scores = self.sample(semmap, prior_latent_space)
        semmap_scores = semmap_scores.squeeze(0)
        
        observed_masks = observed_masks.reshape(map_height, map_width)
        height_map = height_map.reshape(map_height, map_width)

        return semmap_scores, observed_masks, height_map


class SMNet(nn.Module):
    def __init__(self, cfg, device):
        super(SMNet, self).__init__()

        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']
        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = torch.device('cpu')  # cpu

        if mem_update == 'lstm':
            self.rnn = nn.LSTMCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)

        elif mem_update == 'gru':
            self.rnn = nn.GRUCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)
        else:
            raise Exception('{} memory update not supported.'.format(mem_update))


        self.decoder = SemmapDecoder(mem_feat_dim, n_obj_classes)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    
    
    def encode(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        T,C,H,W = features.shape

        mask_inliers = ~mask_outliers

        memory_size = map_height * map_width * self.mem_feat_dim * 4 / 1e9
        if memory_size > 5:
            self.device_mem = torch.device('cpu')
        else:
            self.device_mem = torch.device('cuda')
        self.decoder = self.decoder.to(self.device_mem)

        if self.mem_update == 'lstm':
            state = (torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
                     torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((map_height*map_width), dtype=torch.bool, device=self.device)
        height_map = torch.zeros((map_height*map_width), dtype=torch.float, device=self.device)

        for t in tqdm(range(T)):

            feature = features[t,:,:,:]
            world_to_map = proj_wtm[t,:,:,:]
            inliers = mask_inliers[t,:,:]
            height = heights[t,:,:]

            world_to_map = world_to_map.long()

            feature = feature.to(self.device)
            world_to_map = world_to_map.to(self.device)
            inliers = inliers.to(self.device)
            height = height.to(self.device)


            if self.ego_downsample:
                world_to_map = world_to_map[::4, ::4, :]
                inliers = inliers[::4,::4]
                height = height[::4,::4]

            flat_indices = (map_width*world_to_map[:,:,1] + world_to_map[:,:,0]).long()
            flat_indices = flat_indices[inliers]
            height = height[inliers]
            height += 1000
            height_map, highest_height_indices = scatter_max(
                height,
                flat_indices,
                dim=0,
                out = height_map,
            )

            m = highest_height_indices >= 0

            observed_masks += m

            if m.any():
                feature = F.interpolate(feature.unsqueeze(0), size=(480,640), mode="bilinear", align_corners=True)
                feature = feature.squeeze(0)
                if self.ego_downsample:
                    feature = feature[:,::4,::4]

                feature = feature.permute(1,2,0)  # -- (N,H,W,512)

                feature = feature[inliers, :]

                tmp_memory = feature[highest_height_indices[m], :]

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][m,:].to(self.device),
                                 state[1][m,:].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][m,:] = tmp_state[0].to(self.device_mem)
                    state[1][m,:] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[m,:].to(self.device)

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[m,:] = tmp_state.to(self.device_mem)
                else:
                    raise NotImplementedError


                del tmp_memory
            del feature

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            memory = state
        
        memory = memory.view(map_height, map_width, self.mem_feat_dim)

        memory = memory.permute(2,0,1)
        memory = memory.unsqueeze(0)

        return memory, observed_masks, height_map


    def forward(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        memory, observed_masks, height_map = self.encode(features, 
                                                         proj_wtm, 
                                                         mask_outliers, 
                                                         heights, 
                                                         map_height, 
                                                         map_width)

        semmap_scores = self.decoder(memory)
        semmap_scores = semmap_scores.squeeze(0)

        observed_masks = observed_masks.reshape(map_height, map_width)
        height_map = height_map.reshape(map_height, map_width)

        return semmap_scores, observed_masks, height_map






class SemmapDecoder(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):

        super(SemmapDecoder, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(inplace=True),
                                  )

        self.obj_layer = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                      )

    def forward(self, memory):
        l1 = self.layer(memory)
        out_obj = self.obj_layer(l1)
        return out_obj




class AuxSMNet(nn.Module):
    def __init__(self, cfg, device):
        super(AuxSMNet, self).__init__()

        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']
        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = torch.device('cpu')  # cpu

        if mem_update == 'lstm':
            self.rnn = nn.LSTMCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)

        elif mem_update == 'gru':
            self.rnn = nn.GRUCell(ego_feat_dim, mem_feat_dim, bias=True)

            # change default LSTM initialization
            noise = 0.01
            self.rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_hh)
            self.rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.weight_ih)
            self.rnn.bias_hh.data = torch.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
            self.rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.rnn.bias_ih)
        else:
            raise Exception('{} memory update not supported.'.format(mem_update))


        self.decoder = AuxSemmapDecoder(mem_feat_dim, n_obj_classes)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    
    
    def encode(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        T,C,H,W = features.shape

        mask_inliers = ~mask_outliers

        memory_size = map_height * map_width * self.mem_feat_dim * 4 / 1e9
        if memory_size > 5:
            self.device_mem = torch.device('cpu')
        else:
            self.device_mem = torch.device('cuda')
        self.decoder = self.decoder.to(self.device_mem)

        if self.mem_update == 'lstm':
            state = (torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
                     torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((map_height*map_width,self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((map_height*map_width), dtype=torch.bool, device=self.device)
        height_map = torch.zeros((map_height*map_width), dtype=torch.float, device=self.device)

        for t in tqdm(range(T)):

            feature = features[t,:,:,:]
            world_to_map = proj_wtm[t,:,:,:]
            inliers = mask_inliers[t,:,:]
            height = heights[t,:,:]

            world_to_map = world_to_map.long()

            feature = feature.to(self.device)
            world_to_map = world_to_map.to(self.device)
            inliers = inliers.to(self.device)
            height = height.to(self.device)


            if self.ego_downsample:
                world_to_map = world_to_map[::4, ::4, :]
                inliers = inliers[::4,::4]
                height = height[::4,::4]

            flat_indices = (map_width*world_to_map[:,:,1] + world_to_map[:,:,0]).long()
            flat_indices = flat_indices[inliers]
            height = height[inliers]
            height += 1000
            height_map, highest_height_indices = scatter_max(
                height,
                flat_indices,
                dim=0,
                out = height_map,
            )

            m = highest_height_indices >= 0

            observed_masks += m

            if m.any():
                feature = F.interpolate(feature.unsqueeze(0), size=(480,640), mode="bilinear", align_corners=True)
                feature = feature.squeeze(0)
                if self.ego_downsample:
                    feature = feature[:,::4,::4]

                feature = feature.permute(1,2,0)  # -- (N,H,W,512)

                feature = feature[inliers, :]

                tmp_memory = feature[highest_height_indices[m], :]

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][m,:].to(self.device),
                                 state[1][m,:].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][m,:] = tmp_state[0].to(self.device_mem)
                    state[1][m,:] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[m,:].to(self.device)

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[m,:] = tmp_state.to(self.device_mem)
                else:
                    raise NotImplementedError


                del tmp_memory
            del feature

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            memory = state
        
        memory = memory.view(map_height, map_width, self.mem_feat_dim)

        memory = memory.permute(2,0,1)
        memory = memory.unsqueeze(0)

        return memory, observed_masks, height_map


    def forward(self, features, proj_wtm, mask_outliers, heights, map_height, map_width):

        memory, observed_masks, height_map = self.encode(features, 
                                                         proj_wtm, 
                                                         mask_outliers, 
                                                         heights, 
                                                         map_height, 
                                                         map_width)

        semmap_scores = self.decoder(memory)
        semmap_scores = semmap_scores.squeeze(0)

        observed_masks = observed_masks.reshape(map_height, map_width)
        height_map = height_map.reshape(map_height, map_width)

        return semmap_scores, observed_masks, height_map



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(CNNBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) # kernel size = ...
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if not self.activation:
            return self.batchnorm(self.conv(x))
        return self.relu(self.batchnorm(self.conv(x)))

class Res_block(nn.Module):
    def __init__(self, in_channels, red_channels, out_channels, is_plain=False):
        super(Res_block,self).__init__()
        self.relu = nn.ReLU()
        self.is_plain = is_plain
        
        if in_channels==64:
            self.convseq = nn.Sequential(
                                    CNNBlock(in_channels, red_channels, kernel_size=1, padding=0),
                                    CNNBlock(red_channels, red_channels, kernel_size=3, padding=1),
                                    CNNBlock(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            self.convseq = nn.Sequential(
                                    CNNBlock(in_channels, red_channels, kernel_size=1, padding=0),
                                    CNNBlock(red_channels, red_channels, kernel_size=3, padding=1),
                                    CNNBlock(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Identity()
        else:
            self.convseq = nn.Sequential(
                                    CNNBlock(in_channels, red_channels, kernel_size=1, padding=0, stride=2),
                                    CNNBlock(red_channels, red_channels, kernel_size=3, padding=1),
                                    CNNBlock(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
                
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        
    def forward(self, x):
        y = self.convseq(x)
        if self.is_plain:
            x = y
        else:
            x = y + self.iden(x)
        x = self.relu(x)  # relu(skip connection)
        return x

class AuxSemmapDecoder(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):

        super(AuxSemmapDecoder, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(inplace=True),
                                  )
        self.res_block = Res_block(in_channels=48, red_channels=32, out_channels=48)
        self.obj_layer = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                      )

    def forward(self, memory):
        l1 = self.layer(memory)
        l1_res = self.res_block(l1)
        out_obj = self.obj_layer(l1_res)
        return out_obj