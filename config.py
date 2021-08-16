"""
This file has the information about all the configuration for all the files in this repository.
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # in/out
    parser.add_argument("--outf", default="/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Code/Automated Tumor Segmentation/Experiments", 
        help='trained model will be saved at here')
    parser.add_argument("--save", default="Tumor_Segmentation", 
        help='save name of experiment in args.outf directory')
    parser.add_argument("--path_DICOM_data", default="/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Data/Data_DICOM", 
        help="Path where all the Patient's DICOM data is located")
    parser.add_argument("--path_TIFF_data", default="/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Data/Data_TIFF", 
        help="Path where all the Patient's TIFF data is located")
    parser.add_argument("--annotations", default="T2M", help="Path where all the T2M/T2M+ tiff annotations of each patient are stored")
    # data
    parser.add_argument("--df_train_path", default="/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Code/Automated Tumor Segmentation/save_files/df.csv", help="Path where dataframe_train is saved")
    parser.add_argument('--apply_crop', default=True, help="If True then applies an initial cropping to the entire image slice and mask")
    parser.add_argument('--train_patch_height', default=128)
    parser.add_argument('--train_patch_width', default=128)
    parser.add_argument('--stride_height', default=64)
    parser.add_argument('--stride_width', default=64)
    parser.add_argument('--crop_h', default=4.5, help="fraction of image height that needs to be cropped")
    parser.add_argument('--crop_w', default=4.5, help="fraction of image width that needs to be cropped")
    parser.add_argument('--val_ratio', default=0, help='The ratio of the validation set in the training set')
    # # model parameters
    parser.add_argument('--in_channels', default=1,type=int, help='input channels of model')
    parser.add_argument('--classes', default=2,type=int, help='output channels of model')
    # # training
    parser.add_argument('--N_epochs', default=10000, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--early_stop', default=6, type=int, help='early stopping')
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    # # for pre_trained checkpoint
    # parser.add_argument('--start_epoch', default=1, help='Start epoch')
    # parser.add_argument('--pre_trained', default=None, help='(path of trained _model) load trained model to continue train')
    # testing
    parser.add_argument('--test_patch_height', default=128)
    parser.add_argument('--test_patch_width', default=128)
    # # hardware setting
    parser.add_argument('--cuda', default=True, type=bool, help='Use GPU calculating')
    args = parser.parse_args()
    return args