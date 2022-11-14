import os
from dataset import DataLoaderTrainGoPro, DataLoaderTest

def get_deblur_training_data(rgb_dir, patchsize):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainGoPro(rgb_dir, patchsize, None)

def get_test_data(input_dir):
    return DataLoaderTest(input_dir)

