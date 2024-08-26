import os
import sys

import random
import numpy as np

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from model.network import CoCoReco, ScratchCNN
from loss import MiniBatchLoss
from sklearn.metrics import classification_report, roc_auc_score
import argparse

import time
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter



def parser_args():
    # Create argument parser
    parser = argparse.ArgumentParser(description='CoCoReco Training')

    # Add arguments
    parser.add_argument('--output', type=str, default='output', metavar='DIR', help='output folder')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--is_baselineCNN', action='store_true', default=False, help='use the baseline CNN model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize the input')
    parser.add_argument('--use_causality', action='store_true', default=False, help='use causality maps and factors in the model')
    parser.add_argument('--weight_IT_loss', type=float, default=10, help='weight of the IT loss')
    parser.add_argument('--weight_V1_loss', type=float, default=10, help='weight of the V1 loss')
    parser.add_argument('--weight_PFC_loss', type=float, default=10, help='weight of the PFC loss')
    parser.add_argument('--image_size', type=int, default=128, help='size of input images')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='batch size for validation')
    parser.add_argument('--num_classes', type=int, default=10, help='number of output classes')
    parser.add_argument('--val_interval', type=int, default=5, help='interval of validation epoch')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for data loading: suggested equal to number of CPU cores per task')
    parser.add_argument('--notes', type=str, default=' ', help='notes for the experiment')

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

    global best_Acc
    global best_F1
    global best_AUC 

    # Set the values from the arguments
    num_epochs = args.epochs
    learning_rate = args.lr
    normalize_input = args.normalize
    use_causality = args.use_causality
    weight_IT_loss = args.weight_IT_loss
    weight_V1_loss = args.weight_V1_loss
    weight_PFC_loss = args.weight_PFC_loss
    weight_crossentropy_loss = 1

    image_size = args.image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_classes = args.num_classes
    val_interval = args.val_interval
    workers = args.workers
    notes = args.notes
    
    if rank==0: 
        if args.is_baselineCNN:
            print("Training the baseline CNN model")
            path_to_output_folder = os.path.join(args.output, f"baseCNN_{image_size}_{learning_rate}_{num_epochs}_{train_batch_size}")
        
        else:
            path_to_output_folder = os.path.join(args.output, f"{image_size}_{learning_rate}_{num_epochs}_{train_batch_size}_causal{use_causality}_IT{weight_IT_loss}_V1{weight_V1_loss}_PFC{weight_PFC_loss}")
        
        os.makedirs(path_to_output_folder, exist_ok=True)

    # Set up the logger
    log_file = os.path.join(path_to_output_folder, 'log.txt')
    logger = setup_logger(log_file)   

    # Let us create a Tensorboard writer object to track everything we need and visualize them with TensorboardX
    if rank == 0:
        summary_writer = SummaryWriter(log_dir=path_to_output_folder)
        # For instance, here, we add a string text containing the settings of the experiment
        summary_writer.add_text('Settings',f'num_epochs:{num_epochs},\
                                    learning_rate:{learning_rate},\
                                    normalize_input:{normalize_input},\
                                    image_size:{image_size},\
                                    train_batch_size:{train_batch_size},\
                                    val_batch_size:{val_batch_size},\
                                    num_classes:{num_classes},\
                                    val_interval:{val_interval},\
                                    workers:{workers}')
        summary_writer.add_text('Notes',f'{notes}')
    else:
        summary_writer = None                                    
    
    logger.info(f'MAIN | Distributed init, rank: {rank}, (worldsize: {world_size}), is CUDA available:{torch.cuda.is_available()})')

    if rank==0: logger.info("Starting init_process_group. The main worker is rank 0.")
    
    # Initialize distributed training
    torch.distributed.init_process_group(backend='nccl',
                                        world_size=world_size,
                                        rank=rank)
    if rank==0: logger.info("Init_process_group, done.")

    cudnn.benchmark = True #https://stackoverflow.com/a/58965640    

    local_rank = int(os.environ['SLURM_LOCALID'])

    
    # Define the transformation for the input images. During training, we use Data Augmentation techniques such as RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, and ColorJitter.
    transform_train = transforms.Compose([
        # transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.05),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.ColorJitter(brightness=0.04, contrast=0.04, saturation=0.02, hue=0.01),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val_and_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the Imagenette dataset
    if image_size==160:
        path_to_data = '/leonardo_work/IscrC_CAFE/ECCV2024/imagenette2-160' # curated dataset downscaled to 160x160 preserving aspect ratio
    elif image_size==320:
        path_to_data = '/leonardo_work/IscrC_CAFE/ECCV2024/imagenette2-320' # curated dataset downscaled to 320x320 preserving aspect ratio
    else:
        path_to_data = '/leonardo_work/IscrC_CAFE/ECCV2024/imagenette2' # original size dataset

    # Split the validation dataset into validation (66%) and test (33%) sets
    val_dataset = ImageFolder(os.path.join(path_to_data,'val'), transform=transform_val_and_test)
    val_size = int(0.666 * len(val_dataset))
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [val_size, test_size])
    
    # Load the training dataset
    train_dataset = ImageFolder(os.path.join(path_to_data,'train'), transform=transform_train)

    # Create data samplers for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,shuffle=True, drop_last=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # Create data loaders
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(train_batch_size / world_size), shuffle=False, pin_memory=True, num_workers=workers, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(val_batch_size / world_size), shuffle=False,pin_memory=True, num_workers=workers, sampler=val_sampler, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(train_batch_size / world_size), shuffle=False, pin_memory=True, num_workers=workers, sampler=train_sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(val_batch_size / world_size), shuffle=False, pin_memory=True, num_workers=workers, sampler=test_sampler, drop_last=True)

    # Create the data loaders for training set, validation set, and test set
    logger.info(f"train_loader: {len(train_loader)}")
    logger.info(f"val_loader: {len(val_loader)}")
    logger.info(f"test_loader: {len(test_loader)}")

    # Set the CUDA device based on the local rank
    torch.cuda.set_device(local_rank)
    # Set the device to CUDA
    device = torch.device("cuda")
    # Initialize the model
    if args.is_baselineCNN:
        model = ScratchCNN(num_classes=num_classes, initial_HW_size=image_size).to(device)
    else:
        model = CoCoReco(num_classes=num_classes, initial_HW_size=image_size, use_causality=use_causality).to(device)
    # model = ResNet18(num_classes=num_classes).to(device)

    # Wrap the model in the DistributedDataParallel (DDP) module, providing the ID of available device and ID of output device both to the current (local) rank
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Define loss function for the MiniBatchLoss (imported from loss.py)
    criterion_minibatch = MiniBatchLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    ## Training loop
    # just before training, I want to compute the amount of GPU memory used by the current process 
    logger.info(f"TRAIN | Memory allocated: {torch.cuda.memory_allocated()/1024/1024} MB. Tot: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M trainable params. Starting loop over epochs")

    global loss
    global loss_crossentropy
    global loss_minibatch_it
    global loss_minibatch_v1
    global loss_minibatch_pfc
    
    for epoch in range(num_epochs):       

        use_causality = False
        if not args.is_baselineCNN:
            model.use_causality = False # by default, we do not use causality maps in the model at first epoch (warm-up)        

        if epoch > 0 and args.use_causality and not args.is_baselineCNN:
            use_causality = True
            model.use_causality = use_causality

        # When working with Distributed Data Parallel (DDP), it is important to set the epoch to the sampler because it ensures that each replica of the model uses a different random ordering for each epoch.
        train_sampler.set_epoch(epoch)        
        torch.cuda.empty_cache() 

        model.train()
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # logger.info(f"labels: {labels.shape}")
            # logger.info(f"images: {images.shape}")

            # Forward pass
            # outputs, cmaps_it, cmaps_v1, cmaps_pfc = model(images)  # outputs: the logits, cmaps: the causality maps if use_causality is True, otherwise they are None          
            outputs, cmaps_it, cmaps_v1, cmaps_pfc, _, _, _, _, _, _ = model(images)  # outputs: the logits, cmaps: the causality maps if use_causality is True, otherwise they are None          
            
            # Compute the loss terms
            if use_causality and not args.is_baselineCNN: 
                loss_minibatch_it = criterion_minibatch(cmaps_it, labels)                
                loss_minibatch_v1 = criterion_minibatch(cmaps_v1, labels)
                loss_minibatch_pfc = criterion_minibatch(cmaps_pfc, labels)
            else:
                loss_minibatch_it = 0
                loss_minibatch_v1 = 0
                loss_minibatch_pfc = 0
                # logger.info("No causality maps used in the model, loss_minibatch set to 0")

            loss_crossentropy = criterion(outputs, labels)

            logger.info(f"Loss: {loss_crossentropy}, Loss MiniBatch it: {loss_minibatch_it}, Loss MiniBatch v1: {loss_minibatch_v1}, Loss MiniBatch pfc: {loss_minibatch_pfc}")
            # Aggregate the loss terms
            loss = weight_crossentropy_loss*loss_crossentropy + weight_IT_loss*loss_minibatch_it + weight_V1_loss*loss_minibatch_v1 + weight_PFC_loss*loss_minibatch_pfc
            logger.info(f"Total loss: {loss}")

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

                logger.info(f"VALID | Memory allocated: {torch.cuda.memory_allocated()/1024/1024} MB. Starting loop over val_loader")

                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    # outputs, _, _, _ = model(images)
                    outputs, cmaps_it, cmaps_v1, cmaps_pfc, _, _, _, _, _, _ = model(images) 
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()

                    predicted_labels.extend(predicted.tolist())
                    true_labels.extend(labels.tolist())

                    prob = nn.functional.softmax(outputs, dim=1)
                    predicted_probabilities.extend(prob.tolist())

                # accuracy = 100 * total_correct / total_samples

                report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=['chainsaw','church','englishSpringerDog','frenchHorn','garbageTruck','gasPump','golfBall','musicCassette','parachute','tenchFish'])
                

                accuracy=report['accuracy']
                # precision =  report['macro avg']['precision'] 
                # recall = report['macro avg']['recall']    
                f1_score = report['macro avg']['f1-score']

                # f1_score_weighted = report['weighted avg']['f1-score']

                roc_auc = roc_auc_score(true_labels, predicted_probabilities, multi_class='ovr')   

                ## save to tensorboard
                if dist.get_rank() == 0:
                    summary_writer.add_scalar('Accuracy', accuracy, epoch)
                    summary_writer.add_scalar('F1 Score', f1_score, epoch)
                    # summary_writer.add_scalar('F1 Score weighted', f1_score_weighted, epoch)
                    summary_writer.add_scalar('ROC AUC Score', roc_auc, epoch)    

                # Log the results to print them in the console
                logger.info(f'Epoch [{epoch}/{num_epochs}], Accuracy: {accuracy}, F1 Score: {f1_score},ROC AUC Score (ovr): {roc_auc}')
                # logger.info(f'Epoch [{epoch}/{num_epochs}], F1 Score weighted: {f1_score_weighted}')            

                if (accuracy > best_Acc) and (f1_score > best_F1) and (roc_auc > best_AUC):
                    
                    logger.info(f"{report}")
                    # this is the best global model, achieve the best accuracy, F1 score and ROC AUC score

                    #update the best values
                    best_Acc = accuracy
                    best_epoch = epoch

                    best_F1 = f1_score
                    best_AUC = roc_auc

                    # save the best global model, only rank 0 saves the model
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(), os.path.join(path_to_output_folder, 'bestGlobal.pth.tar'))
                        logger.info(f"Epoch {epoch}: saved best global pth model")
                elif accuracy > best_Acc:                    
                    logger.info(f"{report}")
                    # this is the best accuracy model, but not the best global model (F1 and AUC are not the best). Given our setting, we save the model even in this case
                    best_Acc = accuracy
                    best_epoch = epoch

                    if dist.get_rank() == 0:                        
                        torch.save(model.state_dict(), os.path.join(path_to_output_folder, 'bestAccur.pth.tar'))
                        logger.info(f"Epoch {epoch}: saved pth model with better accuracy")

                
                if best_epoch >= 0 and (epoch - best_epoch) == 8:
                    logger.info("Difference between epoch - best_epoch = {}, trigger rescue strategy!".format(epoch - best_epoch))
                    # Here, we try to avoid early stopping by re-balancing the weights of the loss terms based on the last values of the loss terms
                    total_loss = loss.item()
                    # Cross-entry loss
                    weight_crossentropy_loss = total_loss/(weight_crossentropy_loss*loss_crossentropy.item())
                    # IT loss
                    weight_IT_loss = total_loss/(weight_IT_loss*loss_minibatch_it.item())
                    # V1 loss
                    weight_V1_loss = total_loss/(weight_V1_loss*loss_minibatch_v1.item())
                    # PFC loss
                    weight_PFC_loss = total_loss/(weight_PFC_loss*loss_minibatch_pfc.item())


                if best_epoch >= 0 and (epoch - best_epoch) >= 16:
                    logger.info("Difference between epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                    if dist.get_rank() == 0:
                        filename = sys.argv[0].split(' ')[0].strip()
                        killedlist = kill_process(filename, os.getpid())
                        logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist))
                    break   
    # Cool, we've reached the end!
    logger.info("---End of training---")
    # Just before closing the session, we need to clean the process group in DDP
    dist.destroy_process_group()   

    # When working with multi-node multi-GPU training, it is important to ensure that all processes are terminated correctly. Depending on the environment, you may need to kill the processes manually.
    # Therefore, here, we use an undefined function to force the exit of the processes in case they are not correctly manged before, eventually leading to the halt of the python script due to some error. That is fine.
    if rank==0:
        force_exit_here
    elif rank>0:
        force_exit_here
    print("end")
    return 0            

import logging
def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


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
         args=args,
    )