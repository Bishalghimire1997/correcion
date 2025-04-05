from accelerate import Accelerator
from tqdm.auto import tqdm
import ImageDataset
import model
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
import os

class TrainModel():
    def __init__(self, train_data, target_data):
        mod = Model()
        self.train_data = train_data
        self.target_data = target_data
        self.ds = ImageDataset(train_data, target_data)
        self.train_data_loader = DataLoader(self.ds, batch_size=16, shuffle=True)
        self.model = mod.get_model()
        self.optimizer, self.lr_scheduler = mod.get_cosine_schedule_with_warmup()
        self.noise_scheduler = mod.get_noise_scheduler()

        # Training Parameters
        self.image_size = 128
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.num_epochs = 50
        self.gradient_accumulation_steps = 1
        self.learning_rate = 1e-4
        self.lr_warmup_steps = 500
        self.save_image_epochs = 10
        self.save_model_epochs = 5
        self.mixed_precision = 'fp16'
        self.output_dir = "output"

        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.output_dir, "logs")
        )

    def train(self):
        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            self.accelerator.init_trackers("train_example")

        # Prepare for distributed training
        self.model, self.optimizer, self.train_data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_data_loader, self.lr_scheduler
        )

        global_step = 0

        for epoch in range(self.num_epochs):
            progress_bar = tqdm(total=len(self.train_data_loader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(self.train_data_loader):
                ref_images = batch["reference"]  # Input images with chromatic aberration
                target_images = batch["targets"]  # Ground truth images

                batch_size, channels, height, width = target_images.shape
                target_images_reshaped = target_images.view(batch_size, height * width, channels) # reshaping target images


                # Forward diffusion process
                noise = torch.randn(target_images.shape, device=target_images.device)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (target_images.shape[0],),
                                          device=target_images.device).long()
                noisy_refrences = self.noise_scheduler.add_noise(ref_images, noise, timesteps)




                with self.accelerator.accumulate(self.model):
                    # Model prediction with reference images as conditioning input
                    predicted_image = self.model(
                        noisy_refrences, timesteps,
                        encoder_hidden_states =target_images_reshaped, #Image that are not suffering from chromatic aberations
                        return_dict=False
                    )[0]

                    # Compute loss
                    loss = F.mse_loss(predicted_image, target_images)
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                progress_bar.update(1)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)

                if self.accelerator.is_main_process:
                    self.accelerator.log(logs, step=global_step)
                    global_step += 1

            progress_bar.close()

            # Save model periodically
            if (epoch + 1) % self.save_model_epochs == 0 and self.accelerator.is_main_process:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{epoch+1}.pth"))

            # Optional: Run evaluation after specific epochs
            if (epoch + 1) % self.save_image_epochs == 0:
                self.evaluate(epoch)

    def evaluate(self, epoch):
        """Evaluate the model by generating images from reference images."""
        self.model.eval()
        with torch.no_grad():
            test_dir = os.path.join(self.output_dir, "samples")
            os.makedirs(test_dir, exist_ok=True)

            for step, batch in enumerate(self.train_data_loader):
                ref_images = batch["reference"]
                # Ensure the images are in the expected shape (batch_size, channels, height, width)


                batch_size, channels, height, width = ref_images.shape
                ref_images_reshaped = ref_images.view(batch_size, height * width, channels)

                # Generate images (denoising process)
                generated_images = self.model(
                    ref_images,  # Use random noise for evaluation
                    torch.tensor([0], device=ref_images.device),  # Use t=0 for evaluation
                    encoder_hidden_states=ref_images_reshaped,  # Pass reference images as encoder hidden states
                    return_dict=False)[0]

            # Save the first batch of generated images
                image_grid = make_grid(generated_images, nrow=4)
                image_grid = transforms.ToPILImage()(image_grid)
                image_grid.save(f"{test_dir}/{epoch:04d}.png")
                break
