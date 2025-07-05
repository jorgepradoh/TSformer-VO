import os
import torch
import torch_directml
import torch.optim as optim
from tqdm import tqdm
from datasets.kitti import KITTI
from build_model import build_model
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import pickle
import json
from scripts.poseLossFunction import PoseLoss
from scripts.logger import LossLogger


torch.manual_seed(42)
device = torch_directml.device()

def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Validating ")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            """ if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()
            """
            images, gt = images.to(device), gt.to(device)
            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())

    return epoch_loss / len(val_loader)


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            """if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()"""
            images, gt = images.to(device), gt.to(device)
            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter)

            iter += 1
    return epoch_loss / len(train_loader)  
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    best_val = args["best_val"]
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(args["epoch_init"], epochs):
        # training for one epoch
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args)

        # validate model
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args)

            print(f"Epoch: {epoch} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} \n")

            # save best mode
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_val": best_val,
            }
            if val_loss < best_val:
                print(f"Saving new best model -- loss decreased from {best_val:.6f} to {val_loss:.6f} \n")
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

            # log validation loss in TensorBoard
            tensorboard_writer.add_scalar("val_loss", val_loss, epoch)

        # save checkpoint every 20 epochs
        if not epoch%20:
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))) 
        # save last checkpoint
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))  

        # log loss in TensorBoard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
        sigma_t = torch.exp(criterion.log_sigma_t).item()
        sigma_r = torch.exp(criterion.log_sigma_r).item()
        print("σ_t:", sigma_t, "σ_r:", sigma_r)
        loss_logger.log(epoch, sigma_t, sigma_r)
        
    return

"""
def get_optimizer(params, args):
    method = args["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    return optimizer

    # initialize the optimizer
    if method == "Adam":
        optimizer = optim.Adam(params, lr=args["lr"])
    elif method == "SGD":
        optimizer = optim.SGD(params, lr=args["lr"],
                              momentum=args["momentum"],
                              weight_decay=args["weight_decay"])
    elif method == "RAdam":
        optimizer = optim.RAdam(params, lr=args["lr"])
    elif method == "Adagrad":
        optimizer = optim.Adagrad(params, lr=args["lr"],
                                  weight_decay=args["weight_decay"])

    # load checkpoint
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]), map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
"""

    


def compute_loss(y_hat, y, criterion, args):
    loss = criterion(y_hat, y.float())
    return loss

"""
AdamW decouples weight decay from the gradient update, fixing Adam’s L2-regularization behavior.
It’s now the default optimizer for Transformers (including ViTs) and performs significantly better than vanilla Adam on most vision and regression tasks.

if args["weighted_loss"] == None:
        loss = criterion(y_hat, y.float())
    else:
        y = torch.reshape(y, (y.shape[0], args["window_size"]-1, 6))
        gt_angles = y[:, :, :3].flatten()
        gt_translation = y[:, :, 3:].flatten()

        # predict pose
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], args["window_size"]-1, 6))
        estimated_angles = y_hat[:, :, :3].flatten()
        estimated_translation = y_hat[:, :, 3:].flatten()

        # compute custom loss
        k = args["weighted_loss"]
        loss_angles = k * criterion(estimated_angles, gt_angles.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss =  loss_angles + loss_translation
"""


if __name__ == "__main__":

    # set hyperparameters and configuration
    args = {
        "data_dir": "data",
        "bsize": 4,  # batch size
        "val_split": 0.1,  # percentage to use as validation data
        "window_size": 2,  # number of frames in window
        "overlap": 1,  # number of frames overlapped between windows
#        "optimizer": "AdamW",  # optimizer [Adam, SGD, Adagrad, RAdam]
#        "lr": 1e-5,  # learning rate
        "momentum": 0.9,  # SGD momentum
#        "weight_decay": 1e-4,  # SGD momentum
        "epoch": 100,  # train iters each timestep
    	"weighted_loss": None,  # float to weight angles in loss function
      	"pretrained_ViT": False,  # load weights from pre-trained ViT
        "checkpoint_path": "./checkpoints",  # path to save checkpoint
        "checkpoint": None,  # checkpoint
        "train_split_file": "datasets/train_sequences.txt",
        "val_split_file": "datasets/val_sequences.txt",
    }

    # tiny  - patch_size=16, embed_dim=192, depth=12, num_heads=3
    # small - patch_size=16, embed_dim=384, depth=12, num_heads=6
    # base  - patch_size=16, embed_dim=768, depth=12, num_heads=12
    model_params = {
        "dim": 384,
        "image_size": (192, 640),  #(192, 640),
        "patch_size": 16,
        "attention_type": 'divided_space_time',  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
        "num_frames": args["window_size"],
        "num_classes": 6 * (args["window_size"] - 1),  # 6 DoF for each frame
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }
    args["model_params"] = model_params

    # create checkpoints folder
   # if not os.path.exists(args["checkpoint_path"]):
   #     os.makedirs(args["checkpoint_path"])

    with open(os.path.join(args["checkpoint_path"], 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args["checkpoint_path"], 'args.txt'), 'w') as f:
    	f.write(json.dumps(args))

    # tensorboard writer
    TensorBoardWriter = SummaryWriter(log_dir=args["checkpoint_path"])

    # preprocessing operation
    preprocess = transforms.Compose([
        transforms.Resize((model_params["image_size"])),
        
        # randomly applied data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
        transforms.RandomApply([transforms.RandomAffine(degrees=3, translate=(0.05, 0.05))], p=0.3),


        transforms.ToTensor(),
        transforms.Normalize(
            #mean = [GET S3LI & MADMAX VALUES]
            #std = [GET S3LI & MADMAX VALUES]
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]),
    ])

    # train and val dataloader
    #print("Using CUDA: ", torch.cuda.is_available())
    #device = torch_directml.device()
    print("Using device: ", device)
    print("Loading data...")
    # Load train dataset
    train_data = KITTI(
        window_size=args["window_size"],
        overlap=args["overlap"],
        transform=preprocess,
        sequence_list_file=args["train_split_file"]
    )

    # Load val dataset
    val_data = KITTI(
        window_size=args["window_size"],
        overlap=args["overlap"],
        transform=preprocess,
        sequence_list_file=args["val_split_file"]
    )
  
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args["bsize"],
                                               shuffle=True,
                                               )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             shuffle=False,
                                             )

    # number of samples in each set
    print(f"Train set size: {len(train_data)} samples")
    print(f"Validation set size: {len(val_data)} samples")

    # build and load model
    print("Building model...")
    model, args = build_model(args, model_params)
    model.to(device=device)
    # loss and optimizer
    criterion = PoseLoss()
    loss_logger = LossLogger()
    #optimizer = get_optimizer(model.parameters(), args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # train network
    print(20*"--" +  " Training " + 20*"--")
    train(model, train_loader, val_loader, criterion, optimizer, TensorBoardWriter, args)