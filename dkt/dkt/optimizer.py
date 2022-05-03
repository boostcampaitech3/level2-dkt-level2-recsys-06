from torch.optim import Adam, AdamW, NAdam, Adagrad, Adamax, RAdam, RMSprop, Adadelta, SGD, ASGD, Rprop


def get_optimizer(model, args):
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "adamW":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "NAdam":
        optimizer = NAdam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "Adagrad":
        optimizer = Adagrad(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "Adamax":
        optimizer = Adamax(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "RAdam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "Adadelta":
        optimizer = Adadelta(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "ASGD":
        optimizer = ASGD(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "Rprop":
        optimizer = Rprop(model.parameters(), lr=args.lr)
        
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer
