#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import os
from thop import profile

import numpy as np
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as loader
import torch
import torch.nn.functional as F
import torch.nn as nn
from pycls.core.config import cfg


logger = logging.get_logger(__name__)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
    logger.info("Model:\n{}".format(model))
    # Log model complexity
    # logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    if cfg.TASK == "seg" and cfg.TRAIN.DATASET == "cityscapes":
        h, w = 1025, 2049
    else:
        h, w = cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE
    if cfg.TASK == "jig":
        x = torch.randn(1, cfg.JIGSAW_GRID ** 2, cfg.MODEL.INPUT_CHANNELS, h, w)
    else:
        x = torch.randn(1, cfg.MODEL.INPUT_CHANNELS, h, w)
    macs, params = profile(model, inputs=(x, ), verbose=False)
    logger.info("Params: {:,}".format(params))
    logger.info("Flops: {:,}".format(macs))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
        # Set complexity function to be module's complexity function
        # model.complexity = model.module.complexity
    return model


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Update drop path prob for NAS
    if cfg.MODEL.TYPE == "nas":
        m = model.module if cfg.NUM_GPUS > 1 else model
        m.set_drop_path_prob(cfg.NAS.DROP_PROB * cur_epoch / cfg.OPTIM.MAX_EPOCH)
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate per epoch
    if not cfg.OPTIM.ITER_LR:
        lr = optim.get_epoch_lr(cur_epoch)
        optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Update the learning rate per iter
        if cfg.OPTIM.ITER_LR:
            lr = optim.get_epoch_lr(cur_epoch + cur_iter / len(train_loader))
            optim.set_lr(optimizer, lr)
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        if isinstance(preds, tuple):
            loss = loss_fun(preds[0], labels) + cfg.NAS.AUX_WEIGHT * loss_fun(preds[1], labels)
            preds = preds[0]
        else:
            loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the errors
        if cfg.TASK == "col":
            preds = preds.permute(0, 2, 3, 1)
            preds = preds.reshape(-1, preds.size(3))
            labels = labels.reshape(-1)
            mb_size = inputs.size(0) * inputs.size(2) * inputs.size(3) * cfg.NUM_GPUS
        else:
            mb_size = inputs.size(0) * cfg.NUM_GPUS
        if cfg.TASK == "seg":
            # top1_err is in fact inter; top5_err is in fact union
            top1_err, top5_err = meters.inter_union(preds, labels, cfg.MODEL.NUM_CLASSES)
        else:
            ks = [1, min(5, cfg.MODEL.NUM_CLASSES)]  # rot only has 4 classes
            top1_err, top5_err = meters.topk_errors(preds, labels, ks)
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        if cfg.TASK == "seg":
            top1_err, top5_err = top1_err.cpu().numpy(), top5_err.cpu().numpy()
        else:
            top1_err, top5_err = top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

def train_epoch_pseudo(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Update drop path prob for NAS
    if cfg.MODEL.TYPE == "nas":
        m = model.module if cfg.NUM_GPUS > 1 else model
        m.set_drop_path_prob(cfg.NAS.DROP_PROB * cur_epoch / cfg.OPTIM.MAX_EPOCH)
    # Shuffle the data
    
    # Update the learning rate per epoch
    if not cfg.OPTIM.ITER_LR:
        lr = optim.get_epoch_lr(cur_epoch)
        optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    max_iter=max(len(train_loader[1]),len(train_loader[0]))
    loader.shuffle(train_loader[0], cur_epoch)
    loader.shuffle(train_loader[1], cur_epoch)
    label_iter = iter(train_loader[0])
    unlabel_iter=iter(train_loader[1])
    for cur_iter in range(max_iter):    
        try:
            #print(next(label_iter))
            label_im,_,labels = next(label_iter)
        except:
            loader.shuffle(train_loader[0], cur_epoch)
            label_iter = iter(train_loader[0])
            label_im,_,labels = next(label_iter)
        try:
            unlabel_im1,unlabel_im2,_ = next(unlabel_iter)
        except:
            loader.shuffle(train_loader[1], cur_epoch)
            unlabel_iter = iter(train_loader[1])
            unlabel_im1,unlabel_im2,_ = next(unlabel_iter)
        # Update the learning rate per iter
        if cfg.OPTIM.ITER_LR:
            lr = optim.get_epoch_lr(cur_epoch + cur_iter / max_iter)
            optim.set_lr(optimizer, lr)
        # Transfer the data to the current GPU device
        label_im, labels = label_im.cuda(), labels.cuda(non_blocking=True)
        unlabel_im1, unlabel_im2 = unlabel_im1.cuda(), unlabel_im2.cuda()
        imgs=torch.cat([label_im,unlabel_im1,unlabel_im2],dim=0)
        logits = model(imgs)
        logits_label=logits[:len(labels)]
        logits_unlabel1,logits_unlabel2=torch.split(logits[len(labels):],unlabel_im1.shape[0])

        # with torch.no_grad():
        #     probs = torch.softmax(logits_label, dim=1)
        #     scores, lbs_guess = torch.max(probs, dim=1)
        # print(lbs_guess,labels)
        loss_label=loss_fun(logits_label,labels)
        
        #print(logits.shape,logits_label.shape,logits_unlabel1.shape,logits_unlabel2.shape)

        with torch.no_grad():
            probs = torch.softmax(logits_unlabel1, dim=1)
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(cfg.TRAIN.PSD_THRESHOLD).float()
        criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()
        if cfg.TASK=='psd':
            loss_unlabel=(criteria_u(logits_unlabel1,lbs_u_guess)*mask).mean()
        elif cfg.TASK=='fix':
            loss_unlabel=(criteria_u(logits_unlabel2,lbs_u_guess)*mask).mean()
        else:
            loss_unlabel=0
        
        loss=loss_label+loss_unlabel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the errors
        mb_size = label_im.size(0) * cfg.NUM_GPUS
        ks = [1, min(5, cfg.MODEL.NUM_CLASSES)]  # rot only has 4 classes
        top1_err, top5_err = meters.topk_errors(logits_label, labels, ks)
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss=loss_label.item()
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        
        top1_err, top5_err = top1_err.item(), top5_err.item()
        train_meter.iter_toc() 
        # Update and log stats
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def search_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of differentiable architecture search."""
    m = model.module if cfg.NUM_GPUS > 1 else model
    # Shuffle the data
    loader.shuffle(train_loader[0], cur_epoch)
    loader.shuffle(train_loader[1], cur_epoch)
    # Update the learning rate per epoch
    if not cfg.OPTIM.ITER_LR:
        lr = optim.get_epoch_lr(cur_epoch)
        optim.set_lr(optimizer[0], lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    trainB_iter = iter(train_loader[1])
    for cur_iter, (inputs, labels) in enumerate(train_loader[0]):
        # Update the learning rate per iter
        if cfg.OPTIM.ITER_LR:
            lr = optim.get_epoch_lr(cur_epoch + cur_iter / len(train_loader[0]))
            optim.set_lr(optimizer[0], lr)
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Update architecture
        if cur_epoch + cur_iter / len(train_loader[0]) >= cfg.OPTIM.ARCH_EPOCH:
            try:
                inputsB, labelsB = next(trainB_iter)
            except StopIteration:
                trainB_iter = iter(train_loader[1])
                inputsB, labelsB = next(trainB_iter)
            inputsB, labelsB = inputsB.cuda(), labelsB.cuda(non_blocking=True)
            optimizer[1].zero_grad()
            loss = m._loss(inputsB, labelsB)
            loss.backward()
            optimizer[1].step()
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer[0].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        # Update the parameters
        optimizer[0].step()
        # Compute the errors
        if cfg.TASK == "col":
            preds = preds.permute(0, 2, 3, 1)
            preds = preds.reshape(-1, preds.size(3))
            labels = labels.reshape(-1)
            mb_size = inputs.size(0) * inputs.size(2) * inputs.size(3) * cfg.NUM_GPUS
        else:
            mb_size = inputs.size(0) * cfg.NUM_GPUS
        if cfg.TASK == "seg":
            # top1_err is in fact inter; top5_err is in fact union
            top1_err, top5_err = meters.inter_union(preds, labels, cfg.MODEL.NUM_CLASSES)
        else:
            ks = [1, min(5, cfg.MODEL.NUM_CLASSES)]  # rot only has 4 classes
            top1_err, top5_err = meters.topk_errors(preds, labels, ks)
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        if cfg.TASK == "seg":
            top1_err, top5_err = top1_err.cpu().numpy(), top5_err.cpu().numpy()
        else:
            top1_err, top5_err = top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    # Log genotype
    genotype = m.genotype()
    logger.info("genotype = %s", genotype)
    logger.info(F.softmax(m.net_.alphas_normal, dim=-1))
    logger.info(F.softmax(m.net_.alphas_reduce, dim=-1))


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        if cfg.TASK == "col":
            preds = preds.permute(0, 2, 3, 1)
            preds = preds.reshape(-1, preds.size(3))
            labels = labels.reshape(-1)
            mb_size = inputs.size(0) * inputs.size(2) * inputs.size(3) * cfg.NUM_GPUS
        else:
            mb_size = inputs.size(0) * cfg.NUM_GPUS
        if cfg.TASK == "seg":
            # top1_err is in fact inter; top5_err is in fact union
            top1_err, top5_err = meters.inter_union(preds, labels, cfg.MODEL.NUM_CLASSES)
        else:
            ks = [1, min(5, cfg.MODEL.NUM_CLASSES)]  # rot only has 4 classes
            top1_err, top5_err = meters.topk_errors(preds, labels, ks)
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        if cfg.TASK == "seg":
            top1_err, top5_err = top1_err.cpu().numpy(), top5_err.cpu().numpy()
        else:
            top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(top1_err, top5_err, mb_size)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    result=test_meter.get_epoch_stats(cur_epoch)
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()
    return result

@torch.no_grad()
def test_epoch_semi(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        if cfg.TASK == "col":
            preds = preds.permute(0, 2, 3, 1)
            preds = preds.reshape(-1, preds.size(3))
            labels = labels.reshape(-1)
            mb_size = inputs.size(0) * inputs.size(2) * inputs.size(3) * cfg.NUM_GPUS
        else:
            mb_size = inputs.size(0) * cfg.NUM_GPUS
        if cfg.TASK == "seg":
            # top1_err is in fact inter; top5_err is in fact union
            top1_err, top5_err = meters.inter_union(preds, labels, cfg.MODEL.NUM_CLASSES)
        else:
            ks = [1, min(5, cfg.MODEL.NUM_CLASSES)]  # rot only has 4 classes
            top1_err, top5_err = meters.topk_errors(preds, labels, ks)
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        if cfg.TASK == "seg":
            top1_err, top5_err = top1_err.cpu().numpy(), top5_err.cpu().numpy()
        else:
            top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(top1_err, top5_err, mb_size)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    result=test_meter.get_epoch_stats(cur_epoch)
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()
    return result,result["top1_err"]

def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    if "search" in cfg.MODEL.TYPE:
        params_w = [v for k, v in model.named_parameters() if "alphas" not in k]
        params_a = [v for k, v in model.named_parameters() if "alphas" in k]
        optimizer_w = torch.optim.SGD(
            params=params_w,
            lr=cfg.OPTIM.BASE_LR,
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            dampening=cfg.OPTIM.DAMPENING,
            nesterov=cfg.OPTIM.NESTEROV
        )
        if cfg.OPTIM.ARCH_OPTIM == "adam":
            optimizer_a = torch.optim.Adam(
                params=params_a,
                lr=cfg.OPTIM.ARCH_BASE_LR,
                betas=(0.5, 0.999),
                weight_decay=cfg.OPTIM.ARCH_WEIGHT_DECAY
            )
        elif cfg.OPTIM.ARCH_OPTIM == "sgd":
            optimizer_a = torch.optim.SGD(
                params=params_a,
                lr=cfg.OPTIM.ARCH_BASE_LR,
                momentum=cfg.OPTIM.MOMENTUM,
                weight_decay=cfg.OPTIM.ARCH_WEIGHT_DECAY,
                dampening=cfg.OPTIM.DAMPENING,
                nesterov=cfg.OPTIM.NESTEROV
            )
        optimizer = [optimizer_w, optimizer_a]
    else:
        optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    if cfg.TRAIN.PORTION < 1:
        if "search" in cfg.MODEL.TYPE:
            train_loader = [loader._construct_loader(
                dataset_name=cfg.TRAIN.DATASET,
                split=cfg.TRAIN.SPLIT,
                batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                shuffle=True,
                drop_last=True,
                portion=cfg.TRAIN.PORTION,
                side="l"
            ),
            loader._construct_loader(
                dataset_name=cfg.TRAIN.DATASET,
                split=cfg.TRAIN.SPLIT,
                batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                shuffle=True,
                drop_last=True,
                portion=cfg.TRAIN.PORTION,
                side="r"
            )]
        else:
            train_loader = loader._construct_loader(
                dataset_name=cfg.TRAIN.DATASET,
                split=cfg.TRAIN.SPLIT,
                batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                shuffle=True,
                drop_last=True,
                portion=cfg.TRAIN.PORTION,
                side="l"
            )
        test_loader = loader._construct_loader(
            dataset_name=cfg.TRAIN.DATASET,
            split=cfg.TRAIN.SPLIT,
            batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=False,
            drop_last=False,
            portion=cfg.TRAIN.PORTION,
            side="r"
        )
    elif cfg.TASK=='fix' or cfg.TASK=='psd':
        train_loader = [loader._construct_loader(
                dataset_name=cfg.TRAIN.DATASET,
                split=cfg.TRAIN.SPLIT,
                batch_size=int(cfg.TRAIN.PSD_LABEL_BATCH_SIZE / cfg.NUM_GPUS),
                shuffle=True,
                drop_last=False,
                side="l"
            ),
            loader._construct_loader(
                dataset_name=cfg.TRAIN.DATASET,
                split=cfg.TRAIN.SPLIT,
                batch_size=int(cfg.TRAIN.PSD_UNLABEL_BATCH_SIZE / cfg.NUM_GPUS),
                shuffle=True,
                drop_last=False,
                side="r"
            )]
        test_loader = loader.construct_test_loader()
    else:
        train_loader = loader.construct_train_loader()
        test_loader = loader.construct_test_loader()
    train_meter_type = meters.TrainMeterIoU if cfg.TASK == "seg" else meters.TrainMeter
    test_meter_type = meters.TestMeterIoU if cfg.TASK == "seg" else meters.TestMeter
    if cfg.TASK == 'psd' or cfg.TASK == 'fix':
        l=train_loader[np.argmax([len(train_loader[0]),len(train_loader[1])])]
    else:   
        l = train_loader[0] if isinstance(train_loader, list) else train_loader
    train_meter = train_meter_type(len(l))
    test_meter = test_meter_type(len(test_loader))
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        l = train_loader[0] if isinstance(train_loader, list) else train_loader
        benchmark.compute_time_full(model, loss_fun, l, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        f = search_epoch if "search" in cfg.MODEL.TYPE else train_epoch_pseudo if 'psd' == cfg.TASK or 'fix' == cfg.TASK else train_epoch
        f(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            if cfg.TASK == 'psd' or cfg.TASK == 'fix':
                net.compute_precise_bn_stats(model, train_loader[1])
            else:
                net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, test_meter, cur_epoch)


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()

    # Load model weights
    if cfg.TEST.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
        logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    elif checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint.load_checkpoint(last_checkpoint, model)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
    else:
        print("ERROR: NO checkpoint! ")
        os._exit()
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    if cfg.TASK == 'psd' or cfg.TASK == 'fix':
        result,ce_error=test_epoch_semi(test_loader, model, test_meter, 0)
    else:
        result=test_epoch(test_loader, model, test_meter, 0)
    with open(cfg.OUT_DIR+'/result.txt','w') as f:
        f.write(str(result["top1_err"]))        
    print(result["top1_err"])
    


def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
