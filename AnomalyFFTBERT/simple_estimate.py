import torch
import torch.nn as nn
import numpy as np
import json
import os

from simple_data_setup import create_test_dataloader


def get_scores(dataloader, model, post_activation, window_size, stride, device):
    scores = torch.zeros(size=(dataloader.dataset.get_n_samples(),)).to(device)

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for step, (start_indeces, x, y_true) in enumerate(dataloader):
            n_batch = x.shape[0]
            x, y_true = x.to(device), y_true.to(device)
            y = post_activation(model(x))

            for batch_idx in range(n_batch):
                start_idx = start_indeces[batch_idx]
                end_idx = start_idx + window_size
                # print(f"number of step: {step}, window start idx: {start_idx}")
                scores[start_idx:end_idx] += y[batch_idx]
                # counts[start_idx:end_idx] += 1

        max_overlapping = int(window_size / stride)
        counts = torch.zeros_like(scores)
        for idx in range(0, scores.shape[0], stride):
            counts[idx:idx+window_size] += 1
        for idx in range(scores.shape[0]):
            if counts[idx] < max_overlapping:
                dif = max_overlapping - counts[idx]
                scores[idx] += (scores[idx] * dif)
                counts[idx] = max_overlapping

        scores /= counts

    return scores


def get_y_pred(scores, anomaly_rate):
    q = torch.quantile(scores, torch.Tensor(1 - anomaly_rate).to(scores.device))
    return (scores > q).to(torch.int)


def get_evaluation(scores, dataloader, anomaly_rate_inf, anomaly_rate_sup, anomaly_rate_step):
    labels = dataloader.dataset.get_test_labels()
    best_eval = {"anomaly_rate": 0.,
                 "precision": 0.,
                 "recall": 0.,
                 "f1": 0.,
                 "confusion_matrix": []
                 }
    for anomaly_rate in torch.arange(anomaly_rate_inf, anomaly_rate_sup, anomaly_rate_step):
        y_pred = get_y_pred(scores, anomaly_rate).detach().cpu()

        tp = torch.logical_and(y_pred == 1, labels == 1).sum()
        fp = torch.logical_and(y_pred == 1, labels == 0).sum()
        fn = torch.logical_and(y_pred == 0, labels == 1).sum()
        tn = torch.logical_and(y_pred == 0, labels == 0).sum()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        if f1_score > best_eval["f1"]:
            best_eval["anomaly_rate"] = anomaly_rate
            best_eval["precision"] = precision
            best_eval["recall"] = recall
            best_eval["f1"] = f1_score
            best_eval["confusion_matrix"] = [tp, fp, fn, tn]

    return best_eval


def test(options, model):
    test_data_path = options.test_data_path
    test_label_path = options.test_label_path
    test_dataloader, test_data_info = create_test_dataloader(
        test_data_file=test_data_path,
        test_labels_file=test_label_path,
        window_size=options.window_size,
        stride=options.window_sliding,
        transform=None,
        batch_size=options.batch_size,
        num_workers=1,
        pin_memory=True,
        sub_sequence_rate=1
    )

    print("INFO TEST: Performing inference...")
    sigmoid = nn.Sigmoid().to(options.device)
    test_dataloader.dataset.set_new_test()
    test_scores = get_scores(dataloader=test_dataloader, model=model, post_activation=sigmoid,
                             window_size=options.window_size,
                             stride=options.window_sliding, device=options.device)

    print("INFO TEST: Performing metrics evaluation...")
    best_eval = get_evaluation(scores=test_scores,
                               dataloader=test_dataloader,
                               anomaly_rate_inf=0.001,
                               anomaly_rate_sup=0.301,
                               anomaly_rate_step=0.005)

    print("INFO TEST: Saving test results...")
    np.save(options.result_path, test_scores.cpu().numpy())

    print("INFO TEST: Saving metrics results...")
    best_eval["anomaly_rate"] = best_eval["anomaly_rate"].cpu().numpy().item()
    best_eval["precision"] = best_eval["precision"].cpu().numpy().item()
    best_eval["recall"] = best_eval["recall"].cpu().numpy().item()
    best_eval["f1"] = best_eval["f1"].cpu().numpy().item()

    tp, fp, fn, tn = best_eval["confusion_matrix"]
    best_eval["confusion_matrix"] = [tp.cpu().numpy().item(),
                                     fp.cpu().numpy().item(),
                                     fn.cpu().numpy().item(),
                                     tn.cpu().numpy().item()]

    with open(options.metrics_path, 'w') as f:
        json.dump(best_eval, f, indent=4)
