import os
import configargparse


def setup(config_file_path):
    assert os.path.exists(config_file_path), f"Config file not found at: {config_file_path}"

    parser = configargparse.ArgParser(default_config_files=[config_file_path])

    parser.add("-da",    "--data-folder-A",        type=str,   required=True,      help="directory with aerial image and label data")
    parser.add("-db",    "--data-folder-B",        type=str,   required=True,      help="directory with aerial image and label data")
    parser.add("-hfatr", "--hdf5-file-A-train",    type=str,   required=True,      help="hdf5 file of aerial image and label data")
    parser.add("-hfate", "--hdf5-file-A-test",     type=str,   required=True,      help="hdf5 file of aerial image and label data")
    parser.add("-hfb",   "--hdf5-file-B",          type=str,   required=True,      help="hdf5 file of aerial image and label data")
    parser.add("-re",    "--resume-epoch",         type=str,   default="latest",   help="which checkpoints to load")
    parser.add("-es",    "--epoch-start",          type=int,   default=1,          help="starting epoch number")
    parser.add("-ie",    "--initial-epochs",       type=int,   required=True,      help="number of epochs for training using initial learning rate")
    parser.add("-de",    "--decay-epochs",         type=int,   required=True,      help="number of epochs for training for linearly decaying the learning rate to zero")
    parser.add("-ib",   "--train-batch-size",   type=int,   required=True,      help="batch size for training")
    parser.add("-ls",   "--train-load-size",    type=int,   required=True,      help="load size for training")
    parser.add("-cs",   "--train-crop-size",    type=int,   required=True,      help="crop size for training")
    parser.add("-ic",   "--input-channels",     type=int,   required=True,      help="input channels for training")
    parser.add("-oc",   "--output-channels",    type=int,   required=True,      help="output channels for training")
    parser.add("-o",    "--output-dir",         type=str,   default="output",   help="output dir for TensorBoard and models")
    parser.add("-lr",   "--learning-rate",      type=float, default=2e-4,       help="learning rate of Adam optimizer for training")
    parser.add("-b1",   "--beta1",              type=float, default=0.5,        help="Beta1 value of Adam optimizer for training")
    parser.add("-b2",   "--beta2",              type=float, default=0.999,      help="Beta1 value of Adam optimizer for training")
    parser.add("-gm",   "--gan-mode",           type=str,   default="lsgan",    help="GAN training mode: vanilla, lsgan, wgang")
    parser.add("-dt",   "--dataset-type",       type=str,   default="h5",       help="Custom dataset for .png, .jpg etc.")
    parser.add("-am",   "--augment-mode",       type=str,   default="full",     help="Augmentation mode full or partial.")
    parser.add("-nw",   "--num-workers",        type=int,   default=4,          help="Number of workers for loading the data.")
    parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
    
    conf, _ = parser.parse_known_args()

    return conf

