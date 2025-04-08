import os
import glob
from torch.utils.data import DataLoader

from treemort import dataset
from treemort import sampler
from treemort import augment
from treemort import datautils


def prepare_datasets_h5(conf, train = True):

    if train:
        hdf5_file_path_A = os.path.join(conf.data_folder_A, conf.hdf5_file_A_train)
    else:
        hdf5_file_path_A = os.path.join(conf.data_folder_A, conf.hdf5_file_A_test)
    
    hdf5_file_path_B = os.path.join(conf.data_folder_B, conf.hdf5_file_B)

    image_patch_map_A = datautils.load_and_organize_data(hdf5_file_path_A)
    image_patch_map_B = datautils.load_and_organize_data(hdf5_file_path_B)

    if train:
        # Train loader.
        train_transform = augment.Augmentations_wlabel()
        train_keys_A, _, _ = datautils.stratify_images_by_patch_count(image_patch_map_A, 0, 0)
        train_keys_B, _, _ = datautils.stratify_images_by_patch_count(image_patch_map_B, 0, 0)
        
        load_dataset = dataset.DeadTreeDataset(
                hdf5_file_A=hdf5_file_path_A,
                hdf5_file_B=hdf5_file_path_B,
                keys_A=train_keys_A,
                keys_B=train_keys_B,
                crop_size=conf.train_crop_size,
                transform=train_transform
                )
        loader = DataLoader(load_dataset,
                            batch_size=conf.train_batch_size,
                            sampler=sampler.
                            JointSampler(
                                hdf5_file_path_A, hdf5_file_path_B,
                            train_keys_A, train_keys_B),
                            drop_last=True,
                            num_workers=conf.num_workers)
    else:
        # Test loader.
        _, _, test_keys_A = datautils.stratify_images_by_patch_count(image_patch_map_A, 0, 1)
        _, _, test_keys_B = datautils.stratify_images_by_patch_count(image_patch_map_B, 0, 1)

        load_dataset = dataset.DeadTreeDataset(
                hdf5_file_A=hdf5_file_path_A,
                hdf5_file_B=hdf5_file_path_B,
                keys_A=test_keys_A,
                keys_B=test_keys_B,
                crop=False,
                transform=None
                )
        loader = DataLoader(load_dataset,
                            batch_size=1,
                            sampler=sampler.JointSampler(
                                hdf5_file_path_A,hdf5_file_path_B,
                                test_keys_A, test_keys_B),
                            shuffle=False,
                            drop_last=False,
                            num_workers=conf.num_workers)

    return loader

### Prepare image dataset for the custom A to B tasks.

def prepare_datasets(conf):
    image_dir_A_train = os.path.join(conf.data_folder_A, 'trainA/')
    image_dir_B_train = os.path.join(conf.data_folder_A, 'trainB/')
    image_dir_A_val = os.path.join(conf.data_folder_A, 'valA/')
    image_dir_B_val = os.path.join(conf.data_folder_A, 'valB/')
    if os.path.isdir(image_dir_A_val) == False:
        image_dir_A_val = image_dir_A_train
        image_dir_B_val = image_dir_B_train
    image_dir_A_test = os.path.join(conf.data_folder_A, 'testA/')
    image_dir_B_test = os.path.join(conf.data_folder_A, 'testB/')

    imagename_A_train = glob.glob(os.path.join(image_dir_A_train, '*.' + conf.dataset_type))
    imagename_B_train = glob.glob(os.path.join(image_dir_B_train, '*.' + conf.dataset_type))
    imagename_A_val = glob.glob(os.path.join(image_dir_A_val, '*.' + conf.dataset_type))
    imagename_B_val = glob.glob(os.path.join(image_dir_B_val, '*.' + conf.dataset_type))
    imagename_A_test = glob.glob(os.path.join(image_dir_A_test, '*.' + conf.dataset_type))
    imagename_B_test = glob.glob(os.path.join(image_dir_B_test, '*.' + conf.dataset_type))

    train_transform = augment.Augmentations(train_load_size=conf.train_load_size,
                                            train_crop_size=conf.train_crop_size,
                                            augment_mode=conf.augment_mode)
    val_transform = None
    test_transform = None

    train_dataset = dataset.ImageDataset(
        imagename_A=imagename_A_train,
        imagename_B=imagename_B_train,
        transform=train_transform
    )
    val_dataset = dataset.ImageDataset(
        imagename_A=imagename_A_val,
        imagename_B=imagename_B_val,
        transform=val_transform
    )
    test_dataset = dataset.ImageDataset(
        imagename_A=imagename_A_test,
        imagename_B=imagename_B_test,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset,
                                batch_size=conf.train_batch_size,
                                drop_last=True,
                                shuffle=True,
                                num_workers=conf.num_workers)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False,
                                num_workers=conf.num_workers)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False,
                                num_workers=conf.num_workers)

    return train_loader, val_loader, test_loader