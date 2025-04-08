'''
The implementation of ADA-Net: Attention-Guided Domain Adaptation Network with Contrastive Learning for Standing Dead Tree Segmentation Using Aerial Imagery.
Author: Mete Ahishali,

The software implementation is extensively based on the following repository: https://github.com/taesungp/contrastive-unpaired-translation.
'''
import os
import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt

import adanet
from treemort import config
from treemort import loader

if __name__ == '__main__':

    visualize = True

    conf = config.setup('./configs/adanet.txt')

    model = adanet.AdaNet(conf, train=False)

    if conf.dataset_type == 'h5':
        test_dataset = loader.prepare_datasets_h5(conf, train=False)
    else:
        _, _, test_dataset = loader.prepare_datasets(conf)

    out_dir = os.path.join(conf.output_dir, 'test_epoch_' + conf.resume_epoch)

    os.makedirs(out_dir, exist_ok=True)
    
    if conf.dataset_type == 'h5':
        if os.path.exists(out_dir + '/real_A_test.h5'): os.remove(out_dir + '/real_A_test.h5')
        if os.path.exists(out_dir + '/fake_B_test.h5'): os.remove(out_dir + '/fake_B_test.h5')
        if os.path.exists(out_dir + '/real_B_test.h5'): os.remove(out_dir + '/real_B_test.h5')

    ####################
    ### Test Dataset ###
    ####################
    pbar = tqdm.tqdm(total=len(test_dataset), desc='Testing ')
    for i, data in enumerate(test_dataset):

        model.set_input(data)
        model.test()

        image_index = 0
        real_A = model.real_A[image_index].permute(1, 2, 0).clamp(-1.0, 1.0).detach().cpu().float().numpy()
        fake_B = model.fake_B[image_index].permute(1, 2, 0).clamp(-1.0, 1.0).detach().cpu().float().numpy()
        real_B = model.real_B[image_index].permute(1, 2, 0).clamp(-1.0, 1.0).detach().cpu().float().numpy()

        real_A = (real_A + 1)/2.0
        fake_B = (fake_B + 1)/2.0
        real_B = (real_B + 1)/2.0

        if visualize:
            if conf.dataset_type == 'h5':
                order = [1, 2, 3] # RGB view.
                #order = [0, 1, 2] # NIR-RG view.
                image_name =  os.path.join(out_dir, 'RGB_' + data['key_A'][0] + '.png') # RGB

                _, axs = plt.subplots(1, 3, figsize=(9, 3.5))

                axs[0].imshow(real_A[:, :, order])
                axs[0].axis('off')  # Hide the axes
                axs[0].set_title('Original Image', fontweight='bold')

                axs[1].imshow(fake_B[:, :, order])
                axs[1].axis('off')
                axs[1].set_title('A2G Generated Image', fontweight='bold')

                axs[2].imshow(real_B[:, :, order])
                axs[2].axis('off')
                axs[2].set_title('Real B Image', fontweight='bold')

                plt.tight_layout()
                plt.savefig(image_name)
                plt.close()
            else:
                image_name =  os.path.join(out_dir, data['imagename_A'][0].split('/')[-1]) # RGB

                _, axs = plt.subplots(1, 3, figsize=(9, 3.5))

                axs[0].imshow(real_A)
                axs[0].axis('off')  # Hide the axes
                axs[0].set_title('Original Image', fontweight='bold')

                axs[1].imshow(fake_B)
                axs[1].axis('off')
                axs[1].set_title('A2G Generated Image', fontweight='bold')

                axs[2].imshow(real_B)
                axs[2].axis('off')
                axs[2].set_title('Real B Image', fontweight='bold')

                plt.tight_layout()
                plt.savefig(image_name)
                plt.close()

        if conf.dataset_type == 'h5':
            with h5py.File(out_dir + '/real_A_test.h5', "a") as hf:  # Open in append mode
                
                key = data['key_A'][0]
                hf.create_group(key)
                hf[key].create_dataset("image", data=real_A, compression="gzip")
                hf[key].create_dataset("label", data=np.squeeze(data['label_A'][0].numpy()), compression="gzip")
                hf[key].attrs["contains_dead_tree"] = data['contains_dead_tree_A'][0]
                hf[key].attrs["source_image"] = data['filename_A'][0]

            with h5py.File(out_dir + '/fake_B_test.h5', "a") as hf:  # Open in append mode
        
                key = data['key_A'][0]
                hf.create_group(key)
                hf[key].create_dataset("image", data=fake_B, compression="gzip")
                hf[key].create_dataset("label", data=np.squeeze(data['label_A'][0].numpy()), compression="gzip")
                hf[key].attrs["contains_dead_tree"] = data['contains_dead_tree_A'][0]
                hf[key].attrs["source_image"] = data['filename_A'][0]

            with h5py.File(out_dir + '/real_B_test.h5', "a") as hf:  # Open in append mode
        
                key = data['key_B'][0]
                hf.create_group(key)
                hf[key].create_dataset("image", data=real_B, compression="gzip")
                hf[key].create_dataset("label", data=np.squeeze(data['label_B'][0].numpy()), compression="gzip")
                hf[key].attrs["contains_dead_tree"] = data['contains_dead_tree_B'][0]
                hf[key].attrs["source_image"] = data['filename_B'][0]    
        
        print('Test image processing: ' + str(i + 1))

        pbar.update(1) # Update the iteration number.    
    
    pbar.close()