import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
import pickle

# Import our modules
from diffusion.model import UNetMini
from diffusion.schedulers import DDPMScheduler
from diffusion.sampler import create_sampler
from diffusion.utils import get_device, set_seed, load_checkpoint
from diffusion.sample import get_default_config, load_config, Sampler

class FIDCalculator:
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    """
    
    def __init__(self):
        """
        Initialize FID calculator.
        
        Args:
            device: Device to use
        """
        self.device = get_device()
        
        # Load pre-trained Inception model
        try:
            from torchvision.models import inception_v3
            self.inception_model = inception_v3(pretrained=True, transform_input=False)
            self.inception_model.fc = nn.Identity()  # Remove final classification layer
            self.inception_model.eval()
            self.inception_model.to(self.device)
        except ImportError:
            print("Warning: torchvision not available, using simple feature extractor")
            self.inception_model = self._create_simple_feature_extractor()
    
    def _create_simple_feature_extractor(self):
        """
        Create a simple feature extractor for when Inception is not available.
        """
        model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(self.device)
        return model
    
    def extract_features(self, images):
        """
        Extract features from images using Inception model.
        
        Args:
            images: Batch of images
            
        Returns:
            Feature vectors
        """
        # Resize images to 299x299 for Inception
        if images.shape[1] == 1:  # Grayscale
            images = images.repeat(1, 3, 1, 1)  # Convert to RGB
        
        if images.shape[2] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.inception_model(images)
        
        return features.cpu().numpy()
    
    def calculate_fid(self, real_features, fake_features):
        """
        Calculate FID between real and fake features.
        
        Args:
            real_features: Features from real images
            fake_features: Features from generated images
            
        Returns:
            FID score
        """
        # Calculate mean and covariance
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        covmean = linalg.sqrtm(sigma_real @ sigma_fake)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return fid


class InceptionScoreCalculator:
    """
    Calculate Inception Score for generated images.
    """
    
    def __init__(self):
        """
        Initialize Inception Score calculator.
        
        Args:
            device: Device to use
        """
        self.device = get_device()
        
        # Load pre-trained Inception model
        try:
            from torchvision.models import inception_v3
            self.inception_model = inception_v3(pretrained=True, transform_input=False)
            self.inception_model.eval()
            self.inception_model.to(self.device)
        except ImportError:
            print("Warning: torchvision not available, using simple classifier")
            self.inception_model = self._create_simple_classifier()
    
    def _create_simple_classifier(self):
        """
        Create a simple classifier for when Inception is not available.
        """
        model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),  # 10 classes for MNIST
            nn.Softmax(dim=1)
        ).to(self.device)
        return model
    
    def extract_predictions(self, images):
        """
        Extract class predictions from images.
        
        Args:
            images: Batch of images
            
        Returns:
            Class predictions
        """
        # Resize images to 299x299 for Inception
        if images.shape[1] == 1:  # Grayscale
            images = images.repeat(1, 3, 1, 1)  # Convert to RGB
        
        if images.shape[2] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            predictions = self.inception_model(images)
            if hasattr(self.inception_model, 'fc'):
                predictions = F.softmax(predictions, dim=1)
        
        return predictions.cpu().numpy()
    
    def calculate_inception_score(self, predictions, splits=10):
        """
        Calculate Inception Score.
        
        Args:
            predictions: Class predictions
            splits: Number of splits for calculation
            
        Returns:
            Inception Score
        """
        # Split predictions
        split_size = len(predictions) // splits
        scores = []
        
        for i in range(splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            split_preds = predictions[start_idx:end_idx]
            
            # Calculate mean prediction
            mean_pred = np.mean(split_preds, axis=0)
            
            # Calculate KL divergence
            kl_div = np.sum(split_preds * np.log(split_preds / mean_pred), axis=1)
            scores.append(np.exp(np.mean(kl_div)))
        
        return np.mean(scores), np.std(scores)


class LPIPSCalculator:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity).
    """
    
    def __init__(self):
        """
        Initialize LPIPS calculator.
        
        Args:
            device: Device to use
        """
        self.device = get_device()
        
        # Load LPIPS model
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        except ImportError:
            print("Warning: lpips not available, using simple distance")
            self.lpips_model = None
    
    def calculate_lpips(self, real_images, fake_images):
        """
        Calculate LPIPS between real and fake images.
        
        Args:
            real_images: Real images
            fake_images: Generated images
            
        Returns:
            LPIPS scores
        """
        if self.lpips_model is None:
            # Fallback to simple L2 distance
            return F.mse_loss(real_images, fake_images, reduction='none').mean(dim=[1, 2, 3])
        
        # Ensure images are in the right format for LPIPS
        if real_images.shape[1] == 1:  # Grayscale
            real_images = real_images.repeat(1, 3, 1, 1)
            fake_images = fake_images.repeat(1, 3, 1, 1)
        
        # Normalize to [-1, 1] for LPIPS
        real_images = real_images * 2 - 1
        fake_images = fake_images * 2 - 1
        
        with torch.no_grad():
            lpips_scores = self.lpips_model(real_images, fake_images)
        
        return lpips_scores.squeeze().cpu().numpy()


class PSNRCalculator:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    """
    
    def __init__(self, max_val=1.0):
        """
        Initialize PSNR calculator.
        
        Args:
            max_val: Maximum pixel value
        """
        self.max_val = max_val
    
    def calculate_psnr(self, real_images, fake_images):
        """
        Calculate PSNR between real and fake images.
        
        Args:
            real_images: Real images
            fake_images: Generated images
            
        Returns:
            PSNR scores
        """
        mse = F.mse_loss(real_images, fake_images, reduction='none').mean(dim=[1, 2, 3])
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr.cpu().numpy()


class Evaluator(Sampler):
    """
    Main evaluator class for DDPM models.
    """
    
    def __init__(self, config):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize metric calculators
        self.fid_calculator = FIDCalculator()
        self.is_calculator = InceptionScoreCalculator()
        self.lpips_calculator = LPIPSCalculator()
        self.psnr_calculator = PSNRCalculator()
    
    def create_dataloader(self, split='test'):
        """
        Create dataloader for evaluation.
        
        Args:
            split: Dataset split ('train' or 'test')
            
        Returns:
            DataLoader
        """
        transform = transforms.Compose([
            transforms.Resize(self.config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        dataset = datasets.MNIST(
            root=self.config['data_dir'],
            train=(split == 'train'),
            download=True,
            transform=transform
        )
        
        # Create dataloader using utility function
        from diffusion.utils import create_dataloader
        
        dataloader = create_dataloader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            device=self.device
        )
        
        return dataloader
    
    def evaluate_metrics(self, real_images, fake_images, save_results=True):
        """
        Evaluate all metrics.
        
        Args:
            real_images: Real images
            fake_images: Generated images
            save_results: Whether to save results to file
            
        Returns:
            Dictionary of metrics
        """
        print("Calculating metrics...")
        
        results = {}
        
        # Calculate FID
        print("Calculating FID...")
        real_features = self.fid_calculator.extract_features(real_images)
        fake_features = self.fid_calculator.extract_features(fake_images)
        results['fid'] = self.fid_calculator.calculate_fid(real_features, fake_features)
        
        # Calculate Inception Score
        print("Calculating Inception Score...")
        fake_predictions = self.is_calculator.extract_predictions(fake_images)
        results['inception_score_mean'], results['inception_score_std'] = \
            self.is_calculator.calculate_inception_score(fake_predictions)
        
        # Calculate LPIPS
        print("Calculating LPIPS...")
        results['lpips'] = self.lpips_calculator.calculate_lpips(real_images, fake_images)
        results['lpips_mean'] = np.mean(results['lpips'])
        results['lpips_std'] = np.std(results['lpips'])
        
        # Calculate PSNR
        print("Calculating PSNR...")
        results['psnr'] = self.psnr_calculator.calculate_psnr(real_images, fake_images)
        results['psnr_mean'] = np.mean(results['psnr'])
        results['psnr_std'] = np.std(results['psnr'])
        
        # Print results
        print("\nEvaluation Results:")
        print(f"FID: {results['fid']:.4f}")
        print(f"Inception Score: {results['inception_score_mean']:.4f} ± {results['inception_score_std']:.4f}")
        print(f"LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
        print(f"PSNR: {results['psnr_mean']:.4f} ± {results['psnr_std']:.4f}")
        
        # Save results
        if save_results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
        """
        os.makedirs(self.config['results_dir'], exist_ok=True)
        
        results_file = os.path.join(self.config['results_dir'], 'evaluation_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save as text file
        text_file = os.path.join(self.config['results_dir'], 'evaluation_results.txt')
        with open(text_file, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("==================\n\n")
            f.write(f"FID: {results['fid']:.4f}\n")
            f.write(f"Inception Score: {results['inception_score_mean']:.4f} ± {results['inception_score_std']:.4f}\n")
            f.write(f"LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}\n")
            f.write(f"PSNR: {results['psnr_mean']:.4f} ± {results['psnr_std']:.4f}\n")
        
        print(f"Results saved to {self.config['results_dir']}")
    
    def evaluate(self, num_samples=1000, sampler_type="deterministic", 
                num_inference_steps=50, eta=0.0, save_results=True):
        """
        Main evaluation function.
        
        Args:
            num_samples: Number of samples to evaluate
            sampler_type: Type of sampler
            num_inference_steps: Number of inference steps
            eta: Noise level for DDIM
        """
        print(f"Evaluating model with {num_samples} samples...")
        
        # Load real images
        print("Loading real images...")
        dataloader = self.create_dataloader('test')
        real_images = []
        
        for batch in tqdm(dataloader, desc="Loading real images"):
            images = batch[0].to(self.device)
            real_images.append(images)
            if len(real_images) * self.config['batch_size'] >= num_samples:
                break
        
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        
        # Generate fake images
        print("Generating fake images...")
        fake_images = self.generate_samples(
            num_samples=num_samples,
            sampler_type=sampler_type,
            num_inference_steps=num_inference_steps,
            eta=eta
        )
        
        # Evaluate metrics
        results = self.evaluate_metrics(real_images, fake_images, save_results)
        
        return results


def main():
    """
    Main evaluation function.
    """
    parser = argparse.ArgumentParser(description='Evaluate trained DDPM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--sampler', type=str, default='deterministic', 
                       choices=['ancestral', 'deterministic', 'ddim'], help='Sampler type')
    parser.add_argument('--steps', type=int, default=1000, help='Number of inference steps')
    parser.add_argument('--eta', type=float, default=0.0, help='Noise level for DDIM')
    parser.add_argument('--results-dir', type=str, default='evaluation_results', help='Results directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config: 
        config = load_config(args.config)
    else:
        config = get_default_config()

    config['checkpoint_path'] = args.checkpoint
    config['results_dir'] = args.results_dir
    config['seed'] = args.seed
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Run evaluation
    results = evaluator.evaluate(
        num_samples=args.num_samples,
        sampler_type=args.sampler,
        num_inference_steps=args.steps,
        eta=args.eta,
        save_results=True
    )

    print(results)
    print(f"Results saved to {config['results_dir']}")


if __name__ == "__main__":
    main() 