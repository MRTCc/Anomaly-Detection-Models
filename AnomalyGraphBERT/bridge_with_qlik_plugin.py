import argparse
import numpy as np
from pathlib import Path
import torch

from anomaly_transformer import get_anomaly_transformer
from train_opt import main
from simple_estimate import test as test_model

from custom_argparse import get_parser


def train(args_dict):
    vals = [f"--{k}={v}" for k, v in args_dict.items()]
    parser = get_parser()
    options = parser.parse_args(vals)

    model = main(options)
    return model


def test(args_dict, model):
    vals = [f"--{k}={v}" for k, v in args_dict.items()]
    parser = get_parser()
    options = parser.parse_args(vals)

    test_model(options, model)


def save(model: torch.nn.Module,
         target_dir: str,
         model_name: str):
    """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
    target_dir_path = Path(target_dir)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"INFO SAVE: Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def load(args_dict, model_path: str):
    vals = [f"--{k}={v}" for k, v in args_dict.items()]
    parser = get_parser()
    options = parser.parse_args(vals)

    print("INFO LOAD: Loading data dimensions...")
    data = np.load(options.train_data_path)
    d_data = data.shape[1]
    model = get_anomaly_transformer(input_d_data=d_data,
                                    output_d_data=1 if options.loss == 'bce' else d_data,
                                    patch_size=options.patch_size,
                                    d_embed=options.d_embed,
                                    hidden_dim_rate=4.,
                                    max_seq_len=options.window_size,
                                    positional_encoding=None,
                                    transformer_n_layer=options.n_layer,
                                    dropout=options.dropout)

    print("INFO LOAD: Loading model from .pth/.pt file...")
    model.load_state_dict(torch.load(model_path))

    return model


def predict():
    pass


if __name__ == "__main__":
    args_dict = {
        'train_data_path': r'C:\Users\martu\PycharmProjects\System.AI4AD\datasets\processed\SMAP_train.npy',
        'test_data_path': r'C:\Users\martu\PycharmProjects\System.AI4AD\datasets\processed\SMAP_test.npy',
        'test_label_path': r'C:\Users\martu\PycharmProjects\System.AI4AD\datasets\processed\SMAP_test_label.npy',
        'd_embed': 100,
        'max_steps': 10000,
        'summary_steps': 1000,
        'n_layer': 3,
        'numerical_dims_list': '0',
        'device': 'cuda'
    }

    print("MAIN: training...")
    model = train(args_dict)

    print("MAIN: saving of the trained model...")
    save(model, './logs', 'prova_model.pth')

    print("MAIN: loading of the model...")
    model = load(args_dict, r"C:\Users\martu\PycharmProjects\System.AI4AD\AnomalyFFTBERT\logs\prova_model.pth")

    print("MAIN: testing of the loaded model...")
    test(args_dict, model)
