from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from utils import *
import os


@hydra.main(version_base=None, config_path="config", config_name="config_train_synhsi_trumans")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = cfg.num_gpus

    print('Usable GPUS: ', torch.cuda.device_count(), flush=True)
    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, cfg),
                                nprocs=world_size,
                                join=True)


def train_ddp(rank, world_size, cfg):
    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    guide = list(cfg.guidance.values())[0]

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


    model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)

    synhsi_dataset = hydra.utils.instantiate(cfg.dataset)

    sampler = DistributedSampler(synhsi_dataset)
    dataloader = DataLoader(synhsi_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=True)


    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    trainer.set_dataset_and_model(synhsi_dataset, model)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):

        print(f'Start epoch {epoch}', flush=True)

        sampler.set_epoch(epoch)
        step = 0

        for batch in dataloader:
            step += 1
            optimizer.zero_grad()

            joints, obj_points, mat, scene_flag, action_label = batch
            joints, obj_points, mat, scene_flag, action_label = joints.to(device), obj_points.to(device), \
                                                                mat.to(device), scene_flag.to(device), action_label.to(device)

            t = torch.randint(0, trainer.timesteps, (cfg.batch_size,), device=device).long()
            with torch.no_grad():
                mask, mask_frame, mask_goal = get_mask(joints, guide.mask_ind, fixed_frame=guide.fixed_frame, mask_y=guide.mask_y)
                if not guide.fix_mode:
                    mask = mask_frame

            loss = trainer.p_losses(joints, obj_points, mat, scene_flag, mask, t, action_label)


            if step % 10 == 0:
                print(f"Step: {step} / {len(dataloader)}   Loss: {loss.item()}", flush=True)

            loss.backward()
            optimizer.step()

        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            print(f'Saving checkpoint', flush=True)
            ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"))

        torch.distributed.barrier()

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()


def get_mask(x_start, ind, p=1.0, fixed_frame=0, mask_y=True):
    mask_frame = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)
    mask_goal = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)

    if ind != -1:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_goal[rand_batch, -1, ind * 3: ind * 3 + 3] = True
        if not mask_y:
            mask_goal[rand_batch, -1, ind * 3 + 1] = False

    if fixed_frame > 0:
        mask_frame[:, :fixed_frame, :] = True
    mask = torch.logical_or(mask_frame, mask_goal)
    return mask, mask_frame, mask_goal


if __name__ == '__main__':
    train()
