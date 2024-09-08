import math
import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch import ViT
from tqdm import tqdm
from utils import *


class Sampler:
    def __init__(self, device, mask_ind, emb_f, batch_size, seq_len, channel, fix_mode, timesteps, fixed_frame, **kwargs):
        self.device = device
        self.mask_ind = mask_ind
        self.emb_f = emb_f
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.channel = channel
        self.fix_mode = fix_mode
        self.timesteps = timesteps
        self.fixed_frame = fixed_frame
        self.get_scheduler()

    def set_dataset_and_model(self, dataset, model):
        self.dataset = dataset
        if dataset.load_scene:
            self.grid = dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)
        self.model = model


    def get_scheduler(self):
        betas = linear_beta_schedule(timesteps=self.timesteps)

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.betas = betas

    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def p_losses(self, x_start, obj_points, mat, scene_flag, mask, t, action_label, noise=None, loss_type='huber'):
        if noise is None:
            noise = torch.randn_like(x_start)

        noise[mask] = 0.

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.dataset.load_scene:
            with torch.no_grad():
                x_orig = transform_points(self.dataset.denormalize_torch(x_noisy), mat)
                mat_for_query = mat.clone()
                target_ind = self.mask_ind if self.mask_ind != -1 else 0
                mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
                mat_for_query[:, 1, 3] = 0
                query_points = transform_points(self.grid, mat_for_query)

                occ = self.dataset.get_occ_for_points(query_points, obj_points, scene_flag)
                nb_voxels = self.dataset.nb_voxels
                occ = occ.reshape(-1, nb_voxels, nb_voxels, nb_voxels).float()

                occ = occ.permute(0, 2, 1, 3)
        else:
            occ = None

        predicted_noise = self.model(x_noisy, occ, t, action_label, mask)

        mask_inv = torch.logical_not(mask)

        if loss_type == 'l1':
            loss = F.l1_loss(noise[mask_inv], predicted_noise[mask_inv])
        elif loss_type == 'l2':
            loss = F.mse_loss(noise[mask_inv], predicted_noise[mask_inv])
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise[mask_inv], predicted_noise[mask_inv])
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample_loop(self, fixed_points, obj_points, mat, scene, goal, action_label):
        device = next(self.model.parameters()).device
        shape = (self.batch_size, self.seq_len, self.channel)
        points = torch.randn(shape, device=device)  # + torch.tensor([0., 0.3, 0.] * 22, device=device)

        if self.fix_mode:
            self.set_fixed_points(points, goal, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=True)
        imgs = []
        occs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            model_used = self.model
            points, occ = self.p_sample(model_used, points, fixed_points, goal, obj_points, mat, scene,
                                   torch.full((self.batch_size,), i, device=device, dtype=torch.long), i, action_label, self.mask_ind,
                                   self.emb_f, self.fix_mode)
            if self.fix_mode:
                self.set_fixed_points(points, goal, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=True)

            points_orig = transform_points(self.dataset.denormalize_torch(points), mat)
            imgs.append(points_orig)
            if occ is not None:
                occs.append(occ.cpu().numpy())
        return imgs, occs

    @torch.no_grad()
    def p_sample(self, model, x, fixed_points, goal, obj_points, mat, scene, t, t_index, action_label, mask_ind, emb_f,
                 fix_mode, no_scene=False):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        if self.dataset.load_scene:
            x_orig = transform_points(self.dataset.denormalize_torch(x), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, obj_points, scene)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels, nb_voxels, nb_voxels).float()

            occ = occ.permute(0, 2, 1, 3)

        else:
            occ = None

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, occ, t, action_label, mask=None) / sqrt_one_minus_alphas_cumprod_t
        )
        if not fix_mode:
            self.set_fixed_points(model_mean, goal, fixed_points, mat, joint_id=mask_ind, fix_mode=True, fix_goal=False)

        if t_index == 0:
            return model_mean, occ
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise, occ

        # Algorithm 2 (including returning all images)


    def set_fixed_points(self, img, goal, fixed_points, mat, joint_id=0, fix_mode=False, fix_goal=True):
        # if joint_id != 0:
        #     goal_len = 2
        goal_len = goal.shape[1]
        # goal_batch = goal.reshape(1, 1, 3).repeat(img.shape[0], 1, 1)
        goal = self.dataset.normalize_torch(transform_points(goal, torch.inverse(mat)))
        # img[:, -1, joint_id * 3: joint_id * 3 + 3] = goal_batch[:, 0]
        if fix_goal:
            img[:, -goal_len:, joint_id * 3] = goal[:, :, 0]
            if joint_id != 0:
                img[:, -goal_len:, joint_id * 3 + 1] = goal[:, :, 1]
            img[:, -goal_len:, joint_id * 3 + 2] = goal[:, :, 2]

        if fixed_points is not None and fix_mode:
            img[:, :fixed_points.shape[1], :] = fixed_points


class Unet(nn.Module):
    def __init__(
            self,
            dim_model,
            num_heads,
            num_layers,
            dropout_p,
            dim_input,
            dim_output,
            nb_voxels=None,
            free_p=0.1,
            nb_actions=0,
            ac_type='',
            no_scene=False,
            no_action=False,
            **kwargs
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.nb_actions = nb_actions
        self.ac_type = ac_type
        self.no_scene = no_scene
        self.no_action = no_action

        # LAYERS
        if not no_scene:
            self.scene_embedding = ViT(
                image_size=nb_voxels,
                patch_size=nb_voxels // 4,
                channels=nb_voxels,
                num_classes=dim_model,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
        self.free_p = free_p
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding_input = nn.Linear(dim_input, dim_model)
        self.embedding_output = nn.Linear(dim_output, dim_model)

        # self.embedding_action = nn.Parameter(torch.randn(16, dim_model))

        if not no_action and nb_actions > 0:
            if self.ac_type in ['last_add_first_token', 'last_new_token']:
                self.embedding_action = ActionTransformerEncoder(action_number=nb_actions,
                                                                 dim_model=dim_model,
                                                                 nhead=num_heads // 2,
                                                                 num_layers=num_layers,
                                                                 dim_feedforward=dim_model,
                                                                 dropout_p=dropout_p,
                                                                 activation="gelu")
            elif self.ac_type in ['all_add_token']:
                self.embedding_action = nn.Sequential(
                    nn.Linear(nb_actions, dim_model),
                    nn.SiLU(inplace=False),
                    nn.Linear(dim_model, dim_model),
                )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_model,
                                                   dropout=dropout_p,
                                                   activation="gelu")

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )
        # self.out = nn.Linear(dim_model, dim_output)

        self.out = nn.Linear(dim_model, dim_output)

        self.embed_timestep = TimestepEmbedder(self.dim_model, self.positional_encoder)

    def forward(self, x, cond, timesteps, action, mask, no_action=None):

        #TODO ActionFlag
        # action[action[:, 0] != 0., 0] = 1.

        t_emb = self.embed_timestep(timesteps)  # [1, b, d]

        if self.no_scene:
            scene_emb = torch.zeros_like(t_emb)
        else:
            scene_emb = self.scene_embedding(cond).reshape(-1, 1, self.dim_model)

        if self.no_action or self.nb_actions == 0:
            action_emb = torch.zeros_like(t_emb)
        else:
            if self.ac_type in ['all_add_token']:
                action_emb = self.embedding_action(action)
            elif self.ac_type in ['last_add_first_token', 'last_new_token']:
                action_emb = self.embedding_action(action)
            else:
                raise NotImplementedError

        t_emb = t_emb.permute(1, 0, 2)

        free_ind = torch.rand(scene_emb.shape[0]).to(scene_emb.device) < self.free_p
        scene_emb[free_ind] = 0.
        # if mask is not None:
        #     x[free_ind][:, mask[0]] = 0.

        if self.ac_type in ['last_add_first_token', 'last_new_token']:
            action_emb[free_ind] = 0.
        scene_emb = scene_emb.permute(1, 0, 2)
        action_emb = action_emb.permute(1, 0, 2)

        if self.ac_type in ['all_add_token', 'last_new_token']:
            emb = t_emb + scene_emb
        elif self.ac_type in ['last_add_first_token']:
            emb = t_emb + scene_emb + action_emb

        x = x.permute(1, 0, 2)
        x = self.embedding_input(x) * math.sqrt(self.dim_model)
        if self.ac_type in ['all_add_token', 'last_add_first_token']:
            x = torch.cat((emb, x), dim=0)
        elif self.ac_type in ['last_new_token']:
            x = torch.cat((emb, action_emb, x), dim=0)

        if self.ac_type in ['all_add_token']:
            x[1:] = x[1:] + action_emb

        x = self.positional_encoder(x)
        x = self.transformer(x)
        if self.ac_type in ['all_add_token', 'last_add_first_token']:
            output = self.out(x)[1:]
        elif self.ac_type in ['last_new_token']:
            output = self.out(x)[2:]
        output = output.permute(1, 0, 2)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pos_encoding[timesteps])#.permute(1, 0, 2)


class ActionTransformerEncoder(nn.Module):
    def __init__(self,
                 action_number,
                 dim_model,
                 nhead,
                 num_layers,
                 dim_feedforward,
                 dropout_p,
                 activation="gelu") -> None:
        super().__init__()
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.input_embedder = nn.Linear(action_number, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout_p,
                                                    activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.input_embedder(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1, keepdim=True)
        return x


