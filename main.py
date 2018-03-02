from optparse import OptionParser
from util import *


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input-path", dest="input_path", help="Path to data text files in TFRecord format",
                      default='/projects/xshen/dataset/dailydialog/train')
    parser.add_option("-v", "--validation-dir", dest="validation_dir", help="Path to validation data text files in TFRecord format",
                      default='/projects/xshen/dataset/dailydialog/validation')
    parser.add_option("-t", "--test-dir", dest="test_dir", help="Path to validation data text files in TFRecord format",
                      default='/projects/xshen/dataset/dailydialog/test')
    parser.add_option("--chat", "--chat-test-path", dest="chat_test_path", help="Path to chat test raw data",
                      default='data/test')
    parser.add_option("--wordvec-dict", dest="wvec_dict", help="Path to save word-index dictionary",
                      default='data/Training.dict.pkl')
    parser.add_option("--wordvec-mat", dest="wvec_mat", help="Path to save index-wordvector numpy matrix ",
                      default='/projects/xshen/dataset/dailydialog/embedding.mat')
    parser.add_option("-b", "--batch-size", dest="batch_size", help="Size of mini batch", default=128)

    parser.add_option("--tboard-dir", dest="tboard_dir", help="Directory to log tensorfboard events",
                      default='./Summaries/')
    parser.add_option("--save-path", dest="save_path", help="Path to save checkpoint", default='./Checkpoints/')
    parser.add_option("--save-freq", dest="save_freq", help="Frequency with which to save checkpoint", default=2000)
    parser.add_option("--learning-rate", dest="lr", help="Learning Rate", default=0.0002)
    parser.add_option("--num-epochs", dest="num_epochs", help="Number of epochs", default=20)
    parser.add_option("--hsize", dest="h_size", help="Size of hidden layer in word level", default=512)
    parser.add_option("--csize", dest="c_size", help="Size of hidden layer in sequence-level", default=1024)
    parser.add_option("--zsize", dest="z_size", help="Size of latent variable", default=512)
    parser.add_option("--run-mode", dest="mode", help="0 for train, 1 for test, 2 for test decode word", default=0)
    parser.add_option("--load-chkpt", dest="load_chkpt", help="Path to checkpoint file. Required for mode:1",
                      default='')
    (options, _) = parser.parse_args()
    if int(options.mode) == 0:
        train_with_validate(options)
    elif int(options.mode) == 1:
        test_loss(options)
    elif int(options.mode) == 2:
        chat(options)
