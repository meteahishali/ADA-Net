'''
The implementation of ADA-Net: Attention-Guided Domain Adaptation Network with Contrastive Learning for Standing Dead Tree Segmentation Using Aerial Imagery.
Author: Mete Ahishali,

The software implementation is extensively based on the following repository: https://github.com/taesungp/contrastive-unpaired-translation.
'''
import os
from tqdm import tqdm

import adanet
from treemort import config
from treemort import loader

if __name__ == '__main__':
    conf = config.setup('./configs/adanet.txt')
    
    model = adanet.AdaNet(conf, train=True)
    
    if conf.dataset_type == 'h5':
        train_dataset = loader.prepare_datasets_h5(conf, train=True)
    else:
        train_dataset, val_dataset, _ = loader.prepare_datasets(conf)

    for epoch in range(conf.epoch_start, conf.initial_epochs + conf.decay_epochs + 1):
        print('\nEpoch number is ' + str(epoch))

        pbar = tqdm(total=len(train_dataset), desc='Epoch ' + str(epoch))
        
        for iter_count, data in enumerate(train_dataset):

            model.set_input(data)
            model.optimize_parameters()

            if iter_count % 50 == 0 or iter_count == len(train_dataset) - 1:
                if conf.dataset_type == 'h5':
                    image_name =  os.path.join(conf.output_dir, 'RGB_epoch_' + str(epoch) + '_iteration_' + str(iter_count) + '.png') # RGB
                    model.save_results(image_name, order = [1, 2, 3])
                    image_name =  os.path.join(conf.output_dir, 'NRG_epoch_' + str(epoch) + '_iteration_' + str(iter_count) + '.png') # NIR-RG
                    model.save_results(image_name, order = [0, 1, 2])
                else:
                    image_name =  os.path.join(conf.output_dir, 'epoch_' + str(epoch) + '_iteration_' + str(iter_count) + '.png')
                    model.save_results(image_name, order = [0, 1, 2])

            pbar.update(1) # Update the iteration number.
        
        pbar.close()
        print('\nEnd of the epoch ' + str(epoch))

        model.save_networks('latest')

        if epoch % 5 == 0 or epoch == 1:
            print('Saving the model weights: Epoch '  + str(epoch))
            model.save_networks(epoch)

        model.update_learning_rate()