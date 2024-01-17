import os, time, json
import numpy as np
import torch
import torch.nn as nn
import argparse
import random

from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

import utils.config as config
from AnomalyFFTBERT.simple_data_setup import create_train_dataloader
from AnomalyFFTBERT.simple_data_setup import create_test_dataloader
from AnomalyFFTBERT.simple_estimate import get_scores
from AnomalyFFTBERT.simple_estimate import get_evaluation
from models.anomaly_transformer import get_anomaly_transformer

from AnomalyFFTBERT.simple_data_setup import create_train_epoch_dataloader

from utils.argparse import get_parser

# configuration variables
LOG_DIR = 'logs/'


def transform_to_tensor(src_data):
    return torch.tensor(src_data, dtype=torch.float32)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_default_config(seed):
    set_random_seed(seed)
    # torch.set_default_dtype(torch.float64)


def main(options):
    print("Training script started...")
    set_default_config(seed=0)

    # estrazione delle options
    window_size = options.window_size
    max_seq_len = window_size
    data_seq_len = window_size * options.patch_size

    # N.B. questa è una tupla
    numerical_dims = tuple(int(x) for x in options.numerical_dims_list.split(','))

    transform = transform_to_tensor

    device = torch.device('cuda:{}'.format(options.gpu_id))

    print("Train Dataloader and Test Dataloader instantiation...")
    train_data_path = options.train_data_path
    train_dataloader, train_data_info = create_train_dataloader(
        train_data_file=train_data_path,
        n_windows=options.max_steps * options.batch_size,
        numerical_dims=numerical_dims,
        max_replacing_weight=options.max_replacing_weight,
        flip_replacing_interval=options.flip_replacing_interval,
        white_noising_prob=options.white_noising_prob,
        max_replacing_rate=options.replacing_rate_max,
        window_size=options.window_size,
        uniform_replacing_prob=options.uniform_replacing_prob,
        peak_noising_prob=options.peak_noising_prob,
        length_adjusting_prob=options.length_adjusting_prob,
        soft_replacing_prob=options.soft_replacing_prob,
        transform=transform,
        batch_size=options.batch_size,
        num_workers=1,
        pin_memory=True
    )

    test_data_path = options.test_data_path
    test_label_path = options.test_label_path
    test_dataloader, test_data_info = create_test_dataloader(
        test_data_file=test_data_path,
        test_labels_file=test_label_path,
        window_size=options.window_size,
        stride=options.window_sliding,
        transform=transform,
        batch_size=options.batch_size,
        num_workers=1,
        pin_memory=True,
        sub_sequence_rate=0.5
    )

    d_data = train_data_info["data_dim"]

    print("Model instantiation...")
    model = get_anomaly_transformer(input_d_data=d_data,
                                    output_d_data=1 if options.loss == 'bce' else d_data,
                                    patch_size=options.patch_size,
                                    d_embed=options.d_embed,
                                    hidden_dim_rate=4.,
                                    max_seq_len=window_size,
                                    positional_encoding=None,
                                    transformer_n_layer=options.n_layer,
                                    dropout=options.dropout)

    # print number of learnable parameters of the model (only for debug)
    print(f"Number of learnable parameters of the model: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Load a checkpoint if exists.
    if options.checkpoint is not None:
        print("Loading checkpoints...")
        model.load_state_dict(torch.load(options.checkpoint, map_location='cpu'))

    print("Creating log directory...")
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    log_dir = os.path.join(LOG_DIR, time.strftime('%y%m%d%H%M%S_' + options.dataset, time.localtime(time.time())))
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'state'))

    print("Hyperparameters saving...")
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)

    summary_writer = SummaryWriter(log_dir)
    torch.save(model, os.path.join(log_dir, 'model.pt'))

    print("Loss Function, Optimizer and Scheduler instantiation...")
    lr = options.lr

    if options.loss == 'bce':
        train_loss = nn.BCELoss().to(device)
    else:
        raise SystemError(f"Invalid loss {options.loss}. It is supported only L1 loss!!!")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=options.max_steps,
                                  lr_min=lr * 0.01,
                                  warmup_lr_init=lr * 0.001,
                                  warmup_t=options.max_steps // 10,
                                  cycle_limit=1,
                                  t_in_epochs=False,
                                  )

    print("Starting training...")
    model.to(device)
    model.train()
    sigmoid = nn.Sigmoid().to(device)
    total_val_time = 0.

    for step, (x, y_true) in enumerate(train_dataloader):
        print(f"**** Step {step}")
        start_step_time = time.time()
        x, y_true = x.to(device), y_true.to(device)

        scores = model(x)
        end_forward_time = time.time()

        loss = train_loss(sigmoid(scores), y_true)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip_norm)

        optimizer.step()
        scheduler.step_update(step)

        if step % options.summary_steps == 0:

            model.eval()
            with torch.no_grad():
                if options.loss == 'bce':
                    start_eval_time = time.time()
                    labels = (sigmoid(scores) > 0.5).to(torch.int)
                    accuracy = torch.sum(labels == y_true.to(torch.int)) / (y_true.shape[-1] * y_true.shape[0])
                    summary_writer.add_scalar('Train/Loss', loss.item(), step)
                    summary_writer.add_scalar('Train/Accuracy', accuracy, step)

                    start_score_time = time.time()
                    test_dataloader.dataset.set_new_test()
                    test_scores = get_scores(dataloader=test_dataloader, model=model, post_activation=sigmoid,
                                             window_size=options.window_size,
                                             stride=options.window_sliding, device=device)
                    end_score_time = time.time()

                    test_scores = test_scores.to("cpu")
                    best_eval = get_evaluation(scores=test_scores,
                                               dataloader=test_dataloader,
                                               anomaly_rate_inf=0.001,
                                               anomaly_rate_sup=0.301,
                                               anomaly_rate_step=0.005)

                    end_eval_time = time.time()
                    print(f"test_scores time (s): {end_score_time - start_score_time}")
                    print(f"best_eval time (s): {end_eval_time - end_score_time}")

                    summary_writer.add_scalar('Valid/Evaluation Time', end_eval_time - start_eval_time, step)

                    summary_writer.add_scalar('Valid/Best Anomaly Rate', best_eval["anomaly_rate"], step)
                    summary_writer.add_scalar('Valid/Precision', best_eval["precision"], step)
                    summary_writer.add_scalar('Valid/Recall', best_eval["recall"], step)
                    summary_writer.add_scalar('Valid/F1', best_eval["f1"], step)

                    print(f'iteration: {step} | loss: {loss.item():.10f} | train accuracy: {accuracy:.10f}')
                    print(
                        f'anomaly rate: {best_eval["anomaly_rate"]:.3f} | precision: {best_eval["precision"]:.5f} | '
                        f'recall: {best_eval["recall"]:.5f} | F1-score: {best_eval["f1"]:.5f}\n')

                else:
                    raise SystemError(f"Not valid loss {options.loss}. Only bce loss implemented!!!")
            model.train()

            torch.save(model.state_dict(), os.path.join(log_dir, 'state/state_dict_step_{}.pt'.format(step)))
            print(f"Validation time is: {end_eval_time - start_eval_time:.5f}")

        end_step_time = time.time()
        total_val_time += (end_step_time - start_step_time)
        if step % options.summary_steps == 0:
            print(f"Summary time (s): {total_val_time}")
            summary_writer.add_scalar('Summary Time', total_val_time, step)
            total_val_time = 0
        # TODO: trovare il modo di stampare su stdout il pre-processing time
        summary_writer.add_scalar('Train/Forward Time', end_forward_time - start_step_time, step)
        summary_writer.add_scalar('Train/Step Time', end_step_time - start_step_time, step)

    # Model saving
    # torch.save(model.state_dict(), os.path.join(log_dir, 'state_dict.pt'))

    print("Saving the model...")
    torch.save(model.state_dict(), os.path.join(options.model_save_path, 'state_dict.pt'))

    return model


# second version of the main: the basic idea is to perform an epoch-based training
def second_main(options):
    print("Training script started...")
    set_default_config(seed=0)

    # estrazione delle options
    window_size = options.window_size
    max_seq_len = window_size
    data_seq_len = window_size * options.patch_size

    # N.B. questa è una tupla
    numerical_dims = config.NUMERICAL_COLUMNS[options.dataset]

    transform = transform_to_tensor

    device = torch.device('cuda:{}'.format(options.gpu_id))

    print("Train Dataloader and Test Dataloader instantiation...")
    train_dataloader, train_data_info = create_train_epoch_dataloader(
        train_data_file=r"C:\Users\martu\PycharmProjects\System.AI4AD\datasets\processed\SMAP_train.npy",
        numerical_dims=numerical_dims,
        max_replacing_weight=options.max_replacing_weight,
        flip_replacing_interval=options.flip_replacing_interval,
        white_noising_prob=options.white_noising_prob,
        max_replacing_rate=options.replacing_rate_max,
        window_size=options.window_size,
        uniform_replacing_prob=options.uniform_replacing_prob,
        peak_noising_prob=options.peak_noising_prob,
        length_adjusting_prob=options.length_adjusting_prob,
        soft_replacing_prob=options.soft_replacing_prob,
        transform=transform,
        batch_size=options.batch_size,
        num_workers=1,
        pin_memory=True
    )

    test_dataloader, test_data_info = create_test_dataloader(
        test_data_file=r"C:\Users\martu\PycharmProjects\System.AI4AD\datasets\datasets\data\processed\SMAP_test.npy",
        test_labels_file=r"C:\Users\martu\PycharmProjects\System.AI4AD\datasets\datasets\data\processed\SMAP_test_label.npy",
        window_size=options.window_size,
        stride=options.window_sliding,
        transform=transform,
        batch_size=options.batch_size,
        num_workers=1,
        pin_memory=True
    )

    d_data = train_data_info["data_dim"]

    print("Model instantiation...")
    model = get_anomaly_transformer(input_d_data=d_data,
                                    output_d_data=1 if options.loss == 'bce' else d_data,
                                    patch_size=options.patch_size,
                                    d_embed=options.d_embed,
                                    hidden_dim_rate=4.,
                                    max_seq_len=window_size,
                                    positional_encoding=None,
                                    transformer_n_layer=options.n_layer,
                                    dropout=options.dropout)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Load a checkpoint if exists.
    if options.checkpoint is not None:
        print("Loading checkpoints...")
        model.load_state_dict(torch.load(options.checkpoint, map_location='cpu'))

    print("Creating log directory...")
    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    log_dir = os.path.join(config.LOG_DIR,
                           time.strftime('%y%m%d%H%M%S_' + options.dataset, time.localtime(time.time())))
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'state'))

    print("Hyperparameters saving...")
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)

    summary_writer = SummaryWriter(log_dir)
    torch.save(model, os.path.join(log_dir, 'model.pt'))

    print("Loss Function, Optimizer and Scheduler instantiation...")
    lr = options.lr

    if options.loss == 'bce':
        train_loss = nn.BCELoss().to(device)
    else:
        raise SystemError(f"Invalid loss {options.loss}. It is supported only L1 loss!!!")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=options.max_steps,
                                  lr_min=lr * 0.01,
                                  warmup_lr_init=lr * 0.001,
                                  warmup_t=options.max_steps // 10,
                                  cycle_limit=1,
                                  t_in_epochs=False,
                                  )

    print("Starting training...")
    model.to(device)
    model.train()
    sigmoid = nn.Sigmoid().to(device)
    start_sum_to_sum_time = time.time()
    n_epochs = 10
    for epoch in range(n_epochs):
        print(f"**** Epoch {epoch}")
        start_epoch_time = time.time()

        for step, (x, y_true) in enumerate(train_dataloader):
            print(f"Epoch {epoch}, step {step}")
            x, y_true = x.to(device), y_true.to(device)

            scores = model(x)
            end_forward_time = time.time()

            loss = train_loss(sigmoid(scores), y_true)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip_norm)

            optimizer.step()
            scheduler.step_update(step)
            end_backward_time = time.time()

        start_eval_time = time.time()

        model.eval()
        with torch.no_grad():
            if options.loss == 'bce':
                print(f"Epoch{epoch}: starting evalaution...")
                start_eval_time = time.time()
                labels = (sigmoid(scores) > 0.5).to(torch.int)
                accuracy = torch.sum(labels == y_true.to(torch.int)) / (y_true.shape[-1] * y_true.shape[0])
                summary_writer.add_scalar('Train/Loss', loss.item(), step)
                summary_writer.add_scalar('Train/Accuracy', accuracy, step)

                start_score_time = time.time()
                test_dataloader.dataset.set_new_test()
                test_scores = get_scores(dataloader=test_dataloader, model=model, post_activation=sigmoid,
                                         window_size=options.window_size,
                                         stride=options.window_sliding, device=device)
                start_evaluation_time = time.time()
                print(f"get_scores time (s): {start_evaluation_time - start_score_time:.5f}")
                best_eval = get_evaluation(scores=test_scores,
                                           dataloader=test_dataloader,
                                           anomaly_rate_inf=0.001,
                                           anomaly_rate_sup=0.301,
                                           anomaly_rate_step=0.005)
                end_eval_time = time.time()
                print(f"get_evaluation time (s): {end_eval_time - start_evaluation_time}")
                print(f"Total validation time (s): {end_eval_time - start_eval_time:.5f}")
                summary_writer.add_scalar('Valid/Evaluation Time', end_eval_time - start_eval_time, step)

                summary_writer.add_scalar('Valid/Best Anomaly Rate', best_eval["anomaly_rate"], step)
                summary_writer.add_scalar('Valid/Precision', best_eval["precision"], step)
                summary_writer.add_scalar('Valid/Recall', best_eval["recall"], step)
                summary_writer.add_scalar('Valid/F1', best_eval["f1"], step)

                print(f'iteration: {step} | loss: {loss.item():.10f} | train accuracy: {accuracy:.10f}')
                print(
                    f'anomaly rate: {best_eval["anomaly_rate"]:.3f} | precision: {best_eval["precision"]:.5f} | '
                    f'recall: {best_eval["recall"]:.5f} | F1-score: {best_eval["f1"]:.5f}\n')

            else:
                raise SystemError(f"Not valid loss {options.loss}. Only bce loss implemented!!!")
        model.train()

        torch.save(model.state_dict(), os.path.join(log_dir, 'state/state_dict_step_{}.pt'.format(step)))

        end_epoch_time = time.time()
        print(f"Epoch time: {end_epoch_time - start_epoch_time:.5f} | Validation time: {end_eval_time - start_eval_time:.5f}")


    # Model saving
    torch.save(model.state_dict(), os.path.join(log_dir, 'state_dict.pt'))


if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    main(options)

    # second_main(options)
