from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, LambdaLR, \
                                    StepLR, MultiStepLR, CosineAnnealingLR, CyclicLR, OneCycleLR, \
                                    CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer, args):
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
    elif args.scheduler == "ExponentialLR ":
        print("asdf")
        scheduler = ExponentialLR(optimizer, gamma=0.998)
    #elif args.scheduler
    elif args.scheduler == "LambdaLR":
        scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.95 ** epoch)
    elif args.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)
    elif args.scheduler == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=0.5)
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    elif args.scheduler == "CyclicLR":
        scheduler = CyclicLR(optimizer, base_lr=0.00005, 
                                              step_size_up=5, max_lr=0.0001, 
                                              gamma=0.5, mode='exp_range')
    elif args.scheduler == "OneCycleLR":
        scheduler = OneCycleLR(optimizer, max_lr=0.1, 
                                                steps_per_epoch=10, epochs=10,anneal_strategy='linear')
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, 
                                                                 T_mult=1, eta_min=0.00001)
    return scheduler
