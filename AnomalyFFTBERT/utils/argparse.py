import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # path params
    parser.add_argument("--train_data_path", type=str, help='train data file path (.npy file expected)')
    parser.add_argument("--test_data_path", type=str, help='test data file path (.npy file expected)')
    parser.add_argument("--test_label_path", type=str, help='test labels file path (.npy file expected)')
    parser.add_argument("--model_save_path", type=str, help='path at which '
                                                            'the trained model parameters will be saved')
    parser.add_argument("--result_path", type=str, help='path at which the test result will be saved')
    parser.add_argument("--metrics_path", type=str, help='path at which the evaluated test metrics results'
                                                         ' will be saved')

    # train params
    parser.add_argument("--dataset", default="unknown", type=str, help="name of the dataset used for training")
    parser.add_argument("--device", default="cpu", type=str, help="device on which run the model "
                                                                  "(train/test and inference)")
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--max_steps", default=150000, type=int, help='maximum number of training steps')
    parser.add_argument("--summary_steps", default=500, type=int,
                        help='steps for summarizing and saving of training log')
    parser.add_argument("--checkpoint", default=None, type=str, help='load checkpoint file')
    parser.add_argument("--initial_iter", default=0, type=int, help='initial iteration for training')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--loss", default='bce', type=str, help='loss type')
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)

    # model architecture params
    parser.add_argument("--window_size", default=2048, type=int, help='number of samples for a window')
    parser.add_argument("--patch_size", default=4, type=int, help='number of data points in a patch')
    parser.add_argument("--d_embed", default=512, type=int, help='embedding dimension of feature')
    parser.add_argument("--n_layer", default=6, type=int, help='number of transformer layers')
    parser.add_argument("--dropout", default=0.1, type=float)

    # data degradation params
    parser.add_argument("--numerical_dims_list", default='0', type=str, help='list of numerical '
                                                                              'dims of the dataset')
    parser.add_argument("--soft_replacing_prob", default=0.5, type=float, help='probability for soft replacement')
    parser.add_argument("--uniform_replacing_prob", default=0.15, type=float,
                        help='probability for uniform replacement')
    parser.add_argument("--peak_noising_prob", default=0.15, type=float, help='probability for peak noise')
    parser.add_argument("--length_adjusting_prob", default=0.15, type=float,
                        help='probability for length adjustment')
    parser.add_argument("--white_noising_prob", default=0.05, type=float,
                        help='probability for white noise (deprecated)')
    parser.add_argument("--replacing_rate_max", default=0.15, type=float,
                        help='maximum ratio of replacing interval length to window size')
    parser.add_argument("--flip_replacing_interval", default='all', type=str,
                        help='allowance for random flipping in soft replacement; vertical/horizontal/all/none')
    parser.add_argument("--max_replacing_weight", default=0.7, type=float,
                        help='maximum weight for external interval in soft replacement')

    # test params
    parser.add_argument("--window_sliding", default=16, type=int, help='sliding steps of windows for validation')

    return parser
