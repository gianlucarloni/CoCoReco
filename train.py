import os
import sys

import random
import numpy as np

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from model.network import CoCoReco, ResNet18
from sklearn.metrics import classification_report, roc_auc_score
import argparse

import time
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

def parser_args():
    # Create argument parser
    parser = argparse.ArgumentParser(description='CoCoReco Training')

    # Add arguments
    parser.add_argument('--output', type=str, default='output', metavar='DIR', help='output folder')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize the input')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='batch size for validation')
    parser.add_argument('--num_classes', type=int, default=10, help='number of output classes')
    parser.add_argument('--val_interval', type=int, default=5, help='interval of validation epoch')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for data loading: suggested equal to number of CPU cores per task')

    return parser.parse_args()

def get_args():
    args = parser_args()

    # This will be the output folder
    args.output = os.path.join(os.getcwd(), 'out', args.output, time.strftime("%Y%m%d%H", time.localtime(time.time() + 7200))) # adjust the time zone as needed, such as adding 7200 seconds to the current time.
    return args

def kill_process(filename: str, holdpid: int):
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

best_Acc=0
best_F1=0
best_AUC=0

def main(rank, world_size, args):
    
    print(f'MAIN | Distributed init, rank: {rank}, (worldsize: {world_size}), is CUDA available:{torch.cuda.is_available()})')

    if rank==0: print("Starting init_process_group. The main worker is rank 0.")
    
    # Initialize distributed training
    torch.distributed.init_process_group(backend='nccl',
                                        world_size=world_size,
                                        rank=rank)
    if rank==0: print("Init_process_group, done.")

    cudnn.benchmark = True #https://stackoverflow.com/a/58965640

    if rank==0: 
        os.makedirs(args.output, exist_ok=True)
        print("Output folder created.")

    local_rank = int(os.environ['SLURM_LOCALID'])

    global best_Acc
    global best_F1
    global best_AUC

    # Set the values from the arguments
    num_epochs = args.epochs
    learning_rate = args.lr
    normalize_input = args.normalize
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_classes = args.num_classes
    val_interval = args.val_interval
    workers = args.workers
    
    
    # Define the transformation for the input images
    transform = transforms.Compose([
        transforms.Resize((168, 168)),
        transforms.ToTensor(),
    ])

    if normalize_input:
        transform = transforms.Compose([
            transform,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load the Imagenette dataset
    path_to_data = '/leonardo_work/IscrC_CAFE/ECCV2024/imagenette2-160'
    train_dataset = ImageFolder(os.path.join(path_to_data,'train'), transform=transform)
    val_dataset = ImageFolder(os.path.join(path_to_data,'val'), transform=transform)

    # Create data samplers for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,shuffle=True, drop_last=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(train_batch_size / world_size), shuffle=False, pin_memory=True, num_workers=workers, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(val_batch_size / world_size), shuffle=False,pin_memory=True, num_workers=workers, sampler=val_sampler, drop_last=True)

    # Create the data loaders for training set and validation set
    print(f"train_loader: {len(train_loader)}")
    print(f"val_loader: {len(val_loader)}")




    # Set the CUDA device based on the local rank
    torch.cuda.set_device(local_rank)
    # Set the device to CUDA
    device = torch.device("cuda")
    # Initialize the model
    # model = CoCoReco(num_classes=num_classes).to(device)
    model = ResNet18(num_classes=num_classes).to(device)

    # Wrap the model in the DistributedDataParallel (DDP) module, providing the ID of available device and ID of output device both to the current (local) rank
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):

        # When working with Distributed Data Parallel (DDP), it is important to set the epoch to the sampler because it ensures that each replica of the model uses a different random ordering for each epoch.
        train_sampler.set_epoch(epoch)        
        torch.cuda.empty_cache() 

        model.train()
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # print(f"labels: {labels.shape}")
            # print(f"images: {images.shape}")

            # Forward pass
            outputs = model(images)           
           
            # print(f"outputs: {outputs.shape}")

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Synchronize gradients across all processes
        torch.distributed.barrier()

        # Validation loop
        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                predicted_labels = []
                true_labels = []
                predicted_probabilities = []

                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()

                    predicted_labels.extend(predicted.tolist())
                    true_labels.extend(labels.tolist())

                    prob = nn.functional.softmax(outputs, dim=1)
                    predicted_probabilities.extend(prob.tolist())

                accuracy = 100 * total_correct / total_samples

                report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=['chainsaw','church','englishSpringerDog','frenchHorn','garbageTruck','gasPump','golfBall','musicCassette','parachute','tenchFish'])
                

                accuracy=report['accuracy']
                # precision =  report['macro avg']['precision'] 
                # recall = report['macro avg']['recall']    
                f1_score = report['macro avg']['f1-score']

                roc_auc = roc_auc_score(true_labels, predicted_probabilities, multi_class='ovr')               
                print(f'Epoch [{epoch+1}/{num_epochs}], ROC AUC Score (ovr): {roc_auc}')

                if (accuracy > best_Acc) and (f1_score > best_F1) and (roc_auc > best_AUC):
                    print(f'Epoch [{epoch+1}/{num_epochs}], classification report:\n')
                    print(report)

                    # this is the best global model, achieve the best accuracy, F1 score and ROC AUC score

                    #update the best values
                    best_Acc = accuracy
                    best_epoch = epoch

                    best_F1 = f1_score
                    best_AUC = roc_auc

                    # save the best global model, only rank 0 saves the model
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(), os.path.join(args.output, 'bestGlobal.pth.tar'))
                        print("Saved best global pth model")
                elif accuracy > best_Acc:
                    print(f'Epoch [{epoch+1}/{num_epochs}], classification report:\n')
                    print(report)

                    best_Acc = accuracy
                    if dist.get_rank() == 0:                        
                        torch.save(model.state_dict(), os.path.join(args.output, 'bestAccur.pth.tar'))
                        print("Saved pth model with better accuracy")

                if best_epoch >= 0 and (epoch - best_epoch) >= 8:
                    print("Difference between epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                    if dist.get_rank() == 0:
                        filename = sys.argv[0].split(' ')[0].strip()
                        killedlist = kill_process(filename, os.getpid())
                        print("Kill all process of {}: ".format(filename) + " ".join(killedlist))
                    break   
    # Cool, we've reached the end!
    print("---End of training---")
    # Just before closing the session, we need to clean the process group in DDP
    dist.destroy_process_group()   

    # When working with multi-node multi-GPU training, it is important to ensure that all processes are terminated correctly. Depending on the environment, you may need to kill the processes manually.
    # Therefore, here, we use an undefined function to force the exit of the processes in case they are not correctly manged before, eventually leading to the halt of the python script due to some error. That is fine.
    if rank==0:
        force_exit_here
    elif rank>0:
        force_exit_here

    return 0            

if __name__ == '__main__':
    args = get_args()
    # Set the random seed for reproducibility if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])

    main(rank=rank,
         world_size=world_size,
         args=args
    )