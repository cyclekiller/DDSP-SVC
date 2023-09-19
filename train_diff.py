import os
import argparse
import torch
from torch.optim import lr_scheduler
from logger import utils
from diffusion.data_loaders import get_data_loaders
from diffusion.vocoder import Vocoder, Unit2Mel, Unit2Wav


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    if args.model.type == 'Diffusion':
        from diffusion.solver import train
        model = Unit2Mel(
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans,
                    args.model.n_hidden)
                    
    elif args.model.type == 'DiffusionNew':
        from diffusion.solver_new import train
        model = Unit2Wav(
                args.data.sampling_rate,
                args.data.block_size,
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                pcmer_norm=args.model.pcmer_norm)
        print(' > pcmer_norm:', args.model.pcmer_norm)
                
        if args.discriminator == "formant":
            from formant.formant_discriminator import FormantDiscriminator

            discriminator = FormantDiscriminator()
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer_d = torch.optim.AdamW(discriminator.parameters())
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)

    # load discriminator (if exists)
    try:
        ckpt = torch.load(
            args.env.expdir + "/discriminator.pt", map_location=args.device
        )
        discriminator.combine_discriminator = ckpt["discriminator"]
        optimizer_d = ckpt["optimizer_d"]
    except FileNotFoundError:
        pass

    for param_group in optimizer.param_groups + optimizer_d.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    scheduler_d = lr_scheduler.StepLR(optimizer_d, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    discriminator.to(args.device)

    for optim in [optimizer, optimizer_d]:
        for state in optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    # run
    train(args, initial_global_step, model,
        discriminator, optimizer, optimizer_d, scheduler, scheduler_d,
        vocoder, loader_train, loader_valid)
    
