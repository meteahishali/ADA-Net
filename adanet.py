'''
The implementation of ADA-Net: Attention-Guided Domain Adaptation Network with Contrastive Learning for Standing Dead Tree Segmentation Using Aerial Imagery.
Author: Mete Ahishali,

The software implementation is extensively based on the following repository: https://github.com/taesungp/contrastive-unpaired-translation.
'''
import torch
import os
import matplotlib.pyplot as plt

from models import models, losses

class AdaNet():

    def __init__(self, conf, train=False):

        self.save_dir = os.path.join(conf.output_dir, 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_layers = [0, 4, 8, 12, 16]
        self.feature_dimensions = [conf.input_channels, 128, 256, 256, 256]
        self.freq_patch_size = 32

        self.train = train

        if self.train:
            self.model_names = ['generator', 'pixel_network', 'freq_network', 'discriminator']
        else:
            self.model_names = ['generator']

        self.generator = models.generator(input_channels = conf.input_channels, output_channels = conf.output_channels)
        
        if self.train:
            self.pixel_network = models.MLPs(feature_dimensions = self.feature_dimensions)
            self.freq_network = models.freq_patch(input_channels = conf.input_channels, freq_patch_size = self.freq_patch_size, freq_num_patches=64)
            self.discriminator = models.discriminator(output_channels = conf.output_channels, load_size=conf.train_crop_size)

            # define loss functions
            self.adversarial_criterion = losses.GANLoss(conf.gan_mode).to(self.device)
            self.contrastive_criterion = losses.contrastive_loss(conf.train_batch_size).to(self.device)

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=conf.learning_rate, betas=(conf.beta1, conf.beta2))
            self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=conf.learning_rate, betas=(conf.beta1, conf.beta2))
            self.optimizer_pixel = torch.optim.Adam(self.pixel_network.parameters(), lr=conf.learning_rate, betas=(conf.beta1, conf.beta2))
            self.optimizer_freq = torch.optim.Adam(self.freq_network.parameters(), lr=conf.learning_rate, betas=(conf.beta1, conf.beta2))

            self.optimizers = []
            
            self.optimizers.append(self.optimizer_generator)
            self.optimizers.append(self.optimizer_discriminator)
            self.optimizers.append(self.optimizer_pixel)
            self.optimizers.append(self.optimizer_freq)

        
        if self.train:
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + conf.epoch_start - conf.initial_epochs) / float(conf.decay_epochs + 1)
                return lr_l
            
            self.lr_schedulers = []
            for i in range(0, len(self.optimizers)):
                self.lr_schedulers.append(torch.optim.lr_scheduler.LambdaLR(self.optimizers[i], lr_lambda=lambda_rule))

        self.load_networks(conf.resume_epoch)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (epoch, name)

                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('\nLoading the checkpoint: ' + load_path)
                try:
                    # Try to execute this command
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    net.load_state_dict(state_dict)
                except:
                    print('Checkpoints not found: ' + load_path)
                    print('Loading from scratch.')
                

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    
    def optimize_parameters(self):

        self.forward()

        # Discriminator update.
        self.set_requires_grad(self.discriminator, True)

        self.optimizer_discriminator.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_discriminator.step()

        # Generator update.
        self.set_requires_grad(self.discriminator, False)
        self.optimizer_generator.zero_grad()
        self.optimizer_pixel.zero_grad()
        self.optimizer_freq.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()

        self.optimizer_generator.step()
        self.optimizer_pixel.step()
        self.optimizer_freq.step()


    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.train else self.real_A

        self.fake = self.generator(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        fake = self.fake_B.detach() # No backpropagation to generator by detaching fake_B
        pred_fake = self.discriminator(fake)
        self.loss_D_fake = self.adversarial_criterion(pred_fake, False).mean()

        self.pred_real = self.discriminator(self.real_B)
        loss_D_real = self.adversarial_criterion(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        fake = self.fake_B
        pred_fake = self.discriminator(fake)
        self.loss_G_GAN = self.adversarial_criterion(pred_fake, True).mean()
        self.loss_cont = self.calculate_contrastive_loss(self.real_A, self.fake_B)
        self.loss_cont_idt = self.calculate_contrastive_loss(self.real_B, self.idt_B)
        self.loss_G = self.loss_G_GAN + ((self.loss_cont + self.loss_cont_idt) * 0.5)
        return self.loss_G

    def calculate_contrastive_loss(self, source, target):
        f_query = self.generator(target, self.feature_layers)

        f_key = self.generator(source, self.feature_layers)
        f_key_pool, sample_ids = self.pixel_network(f_key, None)
        f_query_pool, _ = self.pixel_network(f_query, sample_ids)

        freq_k_pool, sample_ids_freq = self.freq_network(source, None)
        freq_q_pool, _ = self.freq_network(target, sample_ids_freq)

        freq_k_pool = torch.stack(freq_k_pool, dim=0)
        freq_q_pool = torch.stack(freq_q_pool, dim=0)
        freq_k_pool = freq_k_pool.flatten(0, 1)
        freq_q_pool = freq_q_pool.flatten(0, 1)

        total_pixel_loss = 0.0
        for i in range(0, len(self.feature_layers)):
            total_pixel_loss += self.contrastive_criterion(f_query_pool[i], f_key_pool[i]).mean()

        loss = self.contrastive_criterion(freq_q_pool, freq_k_pool)
        total_freq_loss = loss.mean()

        total_loss = (total_pixel_loss / len(self.feature_layers)) + (total_freq_loss)

        return total_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def test(self):
        self.generator.eval()
        with torch.no_grad():
            self.forward()

    def save_results(self, image_name, order):
       
        # Save one output from the batch.
        image_index = 0
        real_A = self.real_A[image_index, order, :, :].permute(1, 2, 0).clamp(-1.0, 1.0).detach().cpu().float().numpy()
        fake_B = self.fake_B[image_index, order, :, :].permute(1, 2, 0).clamp(-1.0, 1.0).detach().cpu().float().numpy()
        idt_B = self.idt_B[image_index, order, :, :].permute(1, 2, 0).clamp(-1.0, 1.0).detach().cpu().float().numpy()
        real_B = self.real_B[image_index, order, :, :].permute(1, 2, 0).clamp(-1.0, 1.0).detach().cpu().float().numpy()

        real_A = (real_A + 1)/2.0
        fake_B = (fake_B + 1)/2.0
        idt_B = (idt_B + 1)/2.0
        real_B = (real_B + 1)/2.0

        _, axs = plt.subplots(1, 4, figsize=(12, 3.5))

        axs[0].imshow(real_A)
        axs[0].axis('off')  # Hide the axes
        axs[0].set_title('Original Image', fontweight='bold')

        axs[1].imshow(fake_B)
        axs[1].axis('off')
        axs[1].set_title('A2G Generated Image', fontweight='bold')

        axs[2].imshow(idt_B)
        axs[2].axis('off')
        axs[2].set_title('B2B Identity Image', fontweight='bold')

        axs[3].imshow(real_B)
        axs[3].axis('off')
        axs[3].set_title('Real B Image', fontweight='bold')

        plt.tight_layout()
        plt.savefig(image_name)
        plt.close()

        '''
        image_zero_y = np.ones((real_A.shape[-3], 20, 3))
        image_output_rgb = np.concatenate((image_zero_y, real_A, image_zero_y, fake_B, image_zero_y, idt_B, image_zero_y, real_B, image_zero_y), axis = 1)
        image_zero_x = np.ones((20, image_output_rgb.shape[-2], 3))
        image_output_rgb = np.concatenate((image_zero_x, image_output_rgb, image_zero_x), axis=0)

        image_output_rgb = (255 * image_output_rgb).astype('uint8')

        image_pil_rgb = Image.fromarray(image_output_rgb)
        draw = ImageDraw.Draw(image_pil_rgb)
        draw.text((20, 5), text = 'Original    Image', fill = (0, 0, 0))
        draw.text((40 + 256, 5), text = 'A2B    Generated    Image', fill = (0, 0, 0))
        draw.text((60 + 2*256, 5), text = 'B2B    Identity    Image', fill = (0, 0, 0))
        draw.text((80 + 3*256, 5), text = 'Real    B    Image', fill = (0, 0, 0))

        image_pil_rgb.save(image_name)
        '''
        

    def update_learning_rate(self):

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

