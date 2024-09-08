import torch
        
def predict_noise_conditional(self, x, conditions, timesteps, cfg_scale):
    condition:dict = deepcopy(conditions)
    
    unconditional_condition = deepcopy(conditions)
    batch_size = x.shape[0]
    unconditional_condition.update({'scene_mask':torch.ones([batch_size,]).bool(),
                                    'action_mask':torch.ones([batch_size,]).bool()})
    conditional_noise = self(x, condition, timesteps)
    if cfg_scale == 1:
        return conditional_noise
    else:
        unconditional_noise = self(x, unconditional_condition, timesteps)
        scaled_noise = unconditional_noise + (conditional_noise - unconditional_noise) * cfg_scale 
        return scaled_noise


betas = self._linear_beta_schedule(timesteps=self.timesteps)
self.register_buffer('betas', betas)
# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
self.register_buffer('alphas_cumprod', alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

@torch.no_grad()
def p_sample_classifier_guidance(self, x, t, conditions, 
                                    fix_mask=None, 
                                    fix_values=None, 
                                    sparse_guidance_values=None,
                                    sparse_guidance_mask=None,
                                    cg_scale:float=None,
                                    cfg_scale:float=None,
                                    fix_overwrite_mean:bool=True,
                                    classifier_guidance:bool=True,
                                    cg_loss_type:Literal['L1', 'L2', 'Huber'] = 'L2',):
    assert (t < self.timesteps).all()
    assert (fix_mask is None and fix_values is None) or (fix_mask is not None and fix_values is not None)
    
    sqrt_one_minus_alphas_cumprod_t = self._extract(
        self.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    
    sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
    noise_pred = self.predict_noise_conditional(x,conditions,t,cfg_scale=cfg_scale)
    x_0 = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
    
    x_0 = x_0.clamp(-1,1)
    model_mean = (
        self._extract(self.posterior_mean_coef1, t, x_0.shape) * x_0
        + self._extract(self.posterior_mean_coef2, t, x_0.shape) * x
    ).clamp(-1,1)

    @torch.enable_grad()
    def condition_function(x_t, y):
        x_t = x_t.detach().requires_grad_(True)
        noise_pred = self(x_t, conditions, t)
        x_0 = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        x_0 = x_0.clamp(-1,1)
        if cg_loss_type == 'L1':
            loss = torch.nn.L1Loss()
        elif cg_loss_type == 'L2':
            loss = torch.nn.MSELoss()
        elif cg_loss_type == 'Huber':
            loss = torch.nn.HuberLoss()
        loss = loss(x_0[:, sparse_guidance_mask], y[:, sparse_guidance_mask])
        
        grad = torch.autograd.grad(-loss, x_t)[0]
        # loss_list = torch.sqrt(((x_0[:,sparse_guidance_mask] - y[:, sparse_guidance_mask])**2).sum(dim=1)).float().tolist()
        # for i, number in enumerate(loss_list):
            # loss_list[i] = round(number, 4)

        return grad

    
    if fix_values is not None and fix_overwrite_mean and fix_mask is not None:
        fix_mask = fix_mask.long()
        model_mean = model_mean * (1 - fix_mask) + fix_values * fix_mask
    
    if (t == 0).all():
        if fix_values is not None and fix_overwrite_mean and fix_mask is not None:
            fix_mask = fix_mask.long()
            model_mean = model_mean * (1 - fix_mask) + fix_values * fix_mask
        return {"model_mean": model_mean, "x_t_minus_one": model_mean}
        return model_mean
    else:
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        if cg_scale != 0 and classifier_guidance:
            shift = condition_function(x, sparse_guidance_values)
            print(f'original grad sum: {shift.abs().sum()}, range: {shift.abs().min()} {shift.abs().max()}, average: {shift.abs().mean()}')
            shift = shift * cg_scale * posterior_variance_t
            shift = shift.clamp(-1,1)
            # breakpoint()
            # shift[:, fix_mask] = 0
            print(f'shift sum: {shift.abs().sum()}, range: {shift.abs().min()} {shift.abs().max()}, average: {shift.abs().mean()}')
            print(f'model mean sum: {model_mean.abs().sum()}, range: {model_mean.abs().min()} {model_mean.abs().max()}, average: {model_mean.abs().mean()}')
            model_mean = model_mean + shift
            model_mean = model_mean.clamp(-1,1)

        if exist_nan(model_mean):
            print('nan detected')
            raise ValueError('nan detected')
        
        if fix_values is not None and fix_overwrite_mean and fix_mask is not None:
            fix_mask = fix_mask.long()
            model_mean = model_mean * (1 - fix_mask) + fix_values * fix_mask

        noise = torch.randn_like(x)
        return {"model_mean": model_mean, "x_t_minus_one": model_mean + torch.sqrt(posterior_variance_t) * noise}
        # return model_mean + torch.sqrt(posterior_variance_t) * noise


class CondKeyLocationsWithSdf:
    def __init__(self,
                 target=None,
                 target_mask=None,
                 transform=None,
                 inv_transform=None,
                 abs_3d=False,
                 classifiler_scale=10.0,
                 reward_model=None,
                 reward_model_args=None,
                 use_mse_loss=False,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 use_rand_projection=False,
                 motion_length_cut=6.0,
                 obs_list=[],
                 print_every=None,
                 w_colli=5.0,
                 ):
        self.target = target
        self.target_mask = target_mask
        self.transform = transform
        self.inv_transform = inv_transform
        self.abs_3d = abs_3d
        self.classifiler_scale = classifiler_scale
        self.reward_model = reward_model
        self.reward_model_args = reward_model_args
        self.use_mse_loss = use_mse_loss
        self.guidance_style = guidance_style
        self.stop_cond_from = stop_cond_from
        self.use_rand_projection = use_rand_projection
        self.motion_length_cut = motion_length_cut
        self.cut_frame = int(self.motion_length_cut * 20)
        self.obs_list = obs_list
        self.print_every = print_every

        self.n_joints = 22
        self.w_colli = w_colli
        self.use_smooth_loss = False

    def __call__(self, x, t, p_mean_var, y=None, ):  # *args, **kwds):
        """
        Args:
            target: [bs, 120, 22, 3]
            target_mask: [bs, 120, 22, 3]
        """
        # Stop condition
        if int(t[0]) < self.stop_cond_from:
            return torch.zeros_like(x)
        assert y is not None
        # x shape [bs, 263, 1, 120]
        with torch.enable_grad():
            if self.guidance_style == 'xstart':
                if self.reward_model is not None:
                    # If the reward model is provided, we will use xstart from the
                    # reward model instead.The reward model predict M(x_start | x_t).
                    # The gradient is always computed w.r.t. x_t
                    x = x.detach().requires_grad_(True)
                    reward_model_output = self.reward_model(
                        x, t, **self.reward_model_args)  # this produces xstart
                    xstart_in = reward_model_output
                else:
                    xstart_in = p_mean_var['pred_xstart']
            elif self.guidance_style == 'eps':
                # using epsilon style guidance
                assert self.reward_model is None, "there is no need for the reward model in this case"
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            if y['traj_model']:
                use_rand_proj = False  # x contains only (pose,x,z,y)
            else:
                use_rand_proj = self.use_rand_projection
            x_in_pose_space = self.inv_transform(
                xstart_in.permute(0, 2, 3, 1),
                traject_only=y['traj_model'],
                use_rand_proj=use_rand_proj
            )  # [bs, 1, 120, 263]
            # x_in_adjust[:,:,:, [1,2]] == x_in_joints[:, :, :, 0, [0,2]]
            # Compute (x,y,z) shape [bs, 1, 120, njoints=22, nfeat=3]
            x_in_joints = recover_from_ric(x_in_pose_space, self.n_joints,
                                           abs_3d=self.abs_3d)
            # trajectory is the first joint (pelvis) shape [bs, 120, 3]
            trajec = x_in_joints[:, 0, :, 0, :]
            # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
            # Only care about XZ position for now. Y-axis is going up from the ground
            batch_size = trajec.shape[0]
            trajec = trajec[:, :self.cut_frame, :]
            if self.use_mse_loss:
                loss_kps = F.mse_loss(trajec, self.target[:, :self.cut_frame, 0, :],
                                      reduction='none') * self.target_mask[:, :self.cut_frame, 0, :]
            else:
                loss_kps = F.l1_loss(trajec, self.target[:, :self.cut_frame, 0, :],
                                     reduction='none') * self.target_mask[:, :self.cut_frame, 0, :]
            loss_kps = loss_kps.sum()

            loss_colli = 0.0
            for ((c_x, c_z), rad) in self.obs_list:
                cent = torch.tensor([c_x, c_z], device=trajec.device)
                dist = torch.norm(trajec[:, :, [0, 2]] - cent, dim=2)
                dist = torch.clamp(rad - dist, min=0.0)
                loss_colli += dist.sum() / trajec.shape[1] * self.w_colli

            if self.use_smooth_loss:
                loss_smooth_traj = F.mse_loss(trajec[:, 1:, [0, 2]], trajec[:, :-1, [0, 2]])
                loss_sum += loss_smooth_traj

            loss_kps = loss_kps / self.target_mask.sum() * batch_size
            loss_kps = loss_kps
            loss_sum = loss_kps + loss_colli

            if self.print_every is not None and int(t[0]) % self.print_every == 0:
                print("%03d: %f, %f" % (int(t[0]), float(loss_kps), float(loss_colli)))

            grad = torch.autograd.grad(-loss_sum, x)[0]
            return grad * self.classifiler_scale

