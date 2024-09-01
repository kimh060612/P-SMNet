import torch
import torch.nn as nn
import torch.nn.functional as F
from SMNet.probabilistic_net import AxisAlignedConvGaussian, Fcomb

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

        self.input_channels = ego_feat_dim
        self.num_classes = cfg['n_obj_classes']
        self.num_filters = [32,64,128,192]
        self.latent_dim = 6
        self.no_convs_per_block = 3
        self.no_convs_fcomb = 4
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = 10.
        self.z_prior_sample = 0
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)

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
    

    def encode(self, features, proj_indices, masks_inliers):

        features = features.float()

        N, T, C, H, W = features.shape

        if self.mem_update == 'lstm':
            state = (torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
                     torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
        elif self.mem_update == 'replace':
            state = torch.zeros((N * 250 * 250, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((N, 250, 250), dtype=torch.bool, device=self.device)

        for t in range(T):

            feature = features[:,t,:,:,:]
            mask_inliers = masks_inliers[:,t,:,:]
            proj_index = proj_indices[:,t,:]

            if self.ego_downsample:
                mask_inliers = mask_inliers[:,::4,::4]

            m = (proj_index >= 0)    # -- (N, 250*250)

            if N > 1:
                batch_offset = torch.zeros(N, device=self.device)
                batch_offset[1:] = torch.cumsum(mask_inliers.sum(dim=1).sum(dim=1), dim=0)[:-1]
                batch_offset = batch_offset.unsqueeze(1).repeat(1, 250*250).long()

                proj_index += batch_offset

            if m.any():
                feature = F.interpolate(feature, size=(480, 640), mode="bilinear", align_corners=True)
                if self.ego_downsample:
                    feature = feature[:,:,::4,::4]

                feature = feature.permute(0,2,3,1)  # -- (N,H,W,512)

                feature = feature[mask_inliers, :]
                tmp_memory = feature[proj_index[m], :]

                tmp_top_down_mask = m.view(-1)

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][tmp_top_down_mask,:].to(self.device),
                                 state[1][tmp_top_down_mask,:].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][tmp_top_down_mask,:] = tmp_state[0].to(self.device_mem)
                    state[1][tmp_top_down_mask,:] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[tmp_top_down_mask,:].to(self.device)

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[tmp_top_down_mask,:] = tmp_state.to(self.device_mem)

                elif self.mem_update == 'replace':
                    tmp_memory = self.linlayer(tmp_memory)
                    state[tmp_top_down_mask,:] = tmp_memory.to(self.device_mem)

                else:
                    raise NotImplementedError

                observed_masks += m.reshape(N, 250, 250)

                del tmp_memory
            del feature

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            memory = state
        elif self.mem_update == 'replace':
            memory = state

        memory = memory.view(N, 250, 250, self.mem_feat_dim)

        memory = memory.permute(0,3,1,2)
        memory = memory.to(self.device)

        return memory, observed_masks

    def prior_samples(self, features):
        _, T, _, _, _ = features.shape
        feat_prior_samples = []
        for t in range(T):
            feature = features[:, t, ...]
            sample = self.prior.forward(feature).unsqueeze(1)
            feat_prior_samples.append(sample)
        return torch.mean(torch.cat(feat_prior_samples, dim=1), dim=1)
    
    def posterior_sample(self, features, gt_maps):
        _, T, _, _, _ = features.shape
        feat_posterior_samples = []
        for t in range(T):
            feature = features[:, t, ...]
            gt_map = gt_maps[:, t, ...]
            sample = self.posterior(feature, gt_map).sample().unsqueeze(1)
            feat_posterior_samples.append(sample)
        return torch.mean(torch.cat(feat_posterior_samples, dim=1), dim=1)
        
    def sample(self, map_memory, latent_space):
        return self.fcomb.forward(map_memory, latent_space)
    
    def forward(self, features, proj_indices, masks_inliers, gt_segmap, training=True):
        
        if training:
            posterior_latent_space = self.posterior_sample(features, gt_segmap)
            prior_latent_space = self.prior_samples(features)
            memory, observed_masks = self.encode(features, proj_indices, masks_inliers)
            semmap = self.decoder(memory)

            return self.sample(semmap, prior_latent_space), observed_masks, self.sample(semmap, posterior_latent_space), posterior_latent_space, prior_latent_space
        else:
            prior_latent_space = self.prior_samples(features)
            memory, observed_masks = self.encode(features, proj_indices, masks_inliers)
            semmap = self.decoder(memory)

            return self.sample(semmap, prior_latent_space), observed_masks


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
    

    def encode(self, features, proj_indices, masks_inliers):

        features = features.float()

        N,T,C,H,W = features.shape

        if self.mem_update == 'lstm':
            state = (torch.zeros((N*250*250,self.mem_feat_dim), dtype=torch.float, device=self.device_mem),
                     torch.zeros((N*250*250,self.mem_feat_dim), dtype=torch.float, device=self.device_mem))
        elif self.mem_update == 'gru':
            state = torch.zeros((N*250*250,self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
        elif self.mem_update == 'replace':
            state = torch.zeros((N*250*250,self.mem_feat_dim), dtype=torch.float, device=self.device_mem)

        observed_masks = torch.zeros((N,250,250), dtype=torch.bool, device=self.device)

        for t in range(T):

            feature = features[:,t,:,:,:]
            mask_inliers = masks_inliers[:,t,:,:]
            proj_index = proj_indices[:,t,:]

            if self.ego_downsample:
                mask_inliers = mask_inliers[:,::4,::4]

            m = (proj_index>=0)    # -- (N, 250*250)

            if N > 1:
                batch_offset = torch.zeros(N, device=self.device)
                batch_offset[1:] = torch.cumsum(mask_inliers.sum(dim=1).sum(dim=1), dim=0)[:-1]
                batch_offset = batch_offset.unsqueeze(1).repeat(1, 250*250).long()

                proj_index += batch_offset

            if m.any():
                feature = F.interpolate(feature, size=(480,640), mode="bilinear", align_corners=True)
                if self.ego_downsample:
                    feature = feature[:,:,::4,::4]

                feature = feature.permute(0,2,3,1)  # -- (N,H,W,512)

                feature = feature[mask_inliers, :]

                tmp_memory = feature[proj_index[m], :]

                tmp_top_down_mask = m.view(-1)

                if self.mem_update == 'lstm':
                    tmp_state = (state[0][tmp_top_down_mask,:].to(self.device),
                                 state[1][tmp_top_down_mask,:].to(self.device))

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[0][tmp_top_down_mask,:] = tmp_state[0].to(self.device_mem)
                    state[1][tmp_top_down_mask,:] = tmp_state[1].to(self.device_mem)

                elif self.mem_update == 'gru':
                    tmp_state = state[tmp_top_down_mask,:].to(self.device)

                    tmp_state = self.rnn(tmp_memory, tmp_state)

                    state[tmp_top_down_mask,:] = tmp_state.to(self.device_mem)

                elif self.mem_update == 'replace':
                    tmp_memory = self.linlayer(tmp_memory)
                    state[tmp_top_down_mask,:] = tmp_memory.to(self.device_mem)

                else:
                    raise NotImplementedError

                observed_masks += m.reshape(N,250,250)

                del tmp_memory
            del feature

        if self.mem_update == 'lstm':
            memory = state[0]
        elif self.mem_update == 'gru':
            memory = state
        elif self.mem_update == 'replace':
            memory = state

        memory = memory.view(N,250,250,self.mem_feat_dim)

        memory = memory.permute(0,3,1,2)
        memory = memory.to(self.device)

        return memory, observed_masks


    def forward(self, features, proj_indices, masks_inliers):

        memory, observed_masks = self.encode(features, 
                                             proj_indices, 
                                             masks_inliers)

        semmap = self.decoder(memory)

        return semmap, observed_masks



class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(Conv_block, self).__init__()
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
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Identity()
        else:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0, stride=2),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
                
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
        self.res_block = Res_block(in_channels=48, red_channels=64, out_channels=48)
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






