#!/usr/bin/env python3
"""
MNIST Dataset Visualization Script
Visualizes various aspects of the MNIST dataset to understand the data.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns
from collections import Counter

def load_mnist_data():
    """Load MNIST dataset."""
    print("Loading MNIST dataset...")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Same as your diffusion model
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def visualize_sample_images(dataset, num_samples=16, title="Samples", output_dir="visualize_dataset/MNIST"):
    """Visualize sample images from the dataset."""
    print(f"\nVisualizing {num_samples} sample images...")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(dataloader))
    
    # Denormalize images (convert from [-1, 1] back to [0, 1])
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    # Create grid
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=False, padding=2)
    
    torchvision.utils.save_image(grid, f'{output_dir}/samples.png')
    grid_np = grid.permute(1, 2, 0).numpy()
    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(grid_np)
    plt.title(f"{title}\nLabels: {labels.tolist()}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_class_distribution(dataset, title="Class Distribution", output_dir="visualize_dataset/MNIST"):
    """Visualize the distribution of classes in the dataset."""
    print(f"\nAnalyzing class distribution...")
    
    # Get all labels
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    # Count occurrences
    label_counts = Counter(labels)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    classes = sorted(label_counts.keys())
    counts = [label_counts[c] for c in classes]
    
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy')
    plt.xlabel('Digit Class')
    plt.ylabel('Number of Samples')
    plt.title(f'{title} - Bar Plot')
    plt.xticks(classes)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                str(count), ha='center', va='bottom')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
    plt.title(f'{title} - Pie Chart')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.lower().replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"Class distribution:")
    for digit in sorted(label_counts.keys()):
        count = label_counts[digit]
        percentage = (count / len(dataset)) * 100
        print(f"  Digit {digit}: {count} samples ({percentage:.1f}%)")

def visualize_image_statistics(dataset, num_samples=1000, output_dir="visualize_dataset/MNIST"):
    """Visualize statistics of image pixel values."""
    print(f"\nAnalyzing pixel statistics from {num_samples} samples...")
    
    # Sample images
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, _ = next(iter(dataloader))
    
    # Flatten images
    flat_images = images.view(num_samples, -1)  # Shape: (num_samples, 784)
    
    # Calculate statistics
    mean_pixels = flat_images.mean(dim=0)  # Mean across samples
    std_pixels = flat_images.std(dim=0)    # Std across samples
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # 1. Mean pixel values heatmap
    plt.subplot(2, 3, 1)
    mean_heatmap = mean_pixels.view(32, 32)
    plt.imshow(mean_heatmap, cmap='viridis')
    plt.title('Mean Pixel Values')
    plt.colorbar()
    plt.axis('off')
    
    # 2. Standard deviation heatmap
    plt.subplot(2, 3, 2)
    std_heatmap = std_pixels.view(32, 32)
    plt.imshow(std_heatmap, cmap='plasma')
    plt.title('Pixel Standard Deviation')
    plt.colorbar()
    plt.axis('off')
    
    # 3. Overall pixel value distribution
    plt.subplot(2, 3, 3)
    plt.hist(flat_images.flatten(), bins=50, alpha=0.7, color='blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Overall Pixel Value Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. Sample mean distribution
    plt.subplot(2, 3, 4)
    sample_means = flat_images.mean(dim=1)  # Mean per sample
    plt.hist(sample_means, bins=30, alpha=0.7, color='green')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    plt.title('Sample Mean Distribution')
    plt.grid(True, alpha=0.3)
    
    # 5. Sample standard deviation distribution
    plt.subplot(2, 3, 5)
    sample_stds = flat_images.std(dim=1)  # Std per sample
    plt.hist(sample_stds, bins=30, alpha=0.7, color='red')
    plt.xlabel('Sample Standard Deviation')
    plt.ylabel('Frequency')
    plt.title('Sample Std Distribution')
    plt.grid(True, alpha=0.3)
    
    # 6. Pixel value range per sample
    plt.subplot(2, 3, 6)
    pixel_ranges = flat_images.max(dim=1)[0] - flat_images.min(dim=1)[0]
    plt.hist(pixel_ranges, bins=30, alpha=0.7, color='purple')
    plt.xlabel('Pixel Value Range')
    plt.ylabel('Frequency')
    plt.title('Pixel Value Range per Sample')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pixel_statistics.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"Pixel value statistics:")
    print(f"  Overall mean: {flat_images.mean():.4f}")
    print(f"  Overall std: {flat_images.std():.4f}")
    print(f"  Overall min: {flat_images.min():.4f}")
    print(f"  Overall max: {flat_images.max():.4f}")

def visualize_digit_variations(dataset, num_samples_per_digit=10, output_dir="visualize_dataset/MNIST"):
    """Visualize variations within each digit class."""
    print(f"\nVisualizing digit variations...")
    
    # Create figure
    fig, axes = plt.subplots(10, num_samples_per_digit, figsize=(15, 12))
    
    for digit in range(10):
        # Find samples of this digit
        digit_samples = []
        digit_indices = []
        
        for i in range(len(dataset)):
            if len(digit_samples) >= num_samples_per_digit:
                break
            _, label = dataset[i]
            if label == digit:
                digit_samples.append(i)
        
        # Get images
        for j, idx in enumerate(digit_samples[:num_samples_per_digit]):
            image, _ = dataset[idx]
            
            # Denormalize
            image = (image + 1) / 2
            image = torch.clamp(image, 0, 1)
            
            # Plot
            axes[digit, j].imshow(image.squeeze(), cmap='gray')
            axes[digit, j].axis('off')
            
            if j == 0:
                axes[digit, j].set_ylabel(f'Digit {digit}', fontsize=12)
    
    plt.suptitle(f'MNIST Digit Variations ({num_samples_per_digit} samples per digit)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/digit_variations.png", dpi=150, bbox_inches='tight')
    plt.show()

def visualize_training_vs_test(dataset_train, dataset_test, output_dir="visualize_dataset/MNIST"):
    """Compare training and test set characteristics."""
    print(f"\nComparing training and test sets...")
    
    # Sample from both datasets
    train_loader = DataLoader(dataset_train, batch_size=1000, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=True)
    
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))
    
    # Flatten images
    train_flat = train_images.view(1000, -1)
    test_flat = test_images.view(1000, -1)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # 1. Pixel value distributions
    plt.subplot(2, 3, 1)
    plt.hist(train_flat.flatten(), bins=50, alpha=0.7, label='Training', color='blue')
    plt.hist(test_flat.flatten(), bins=50, alpha=0.7, label='Test', color='red')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Pixel Value Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Sample mean distributions
    plt.subplot(2, 3, 2)
    train_means = train_flat.mean(dim=1)
    test_means = test_flat.mean(dim=1)
    plt.hist(train_means, bins=30, alpha=0.7, label='Training', color='blue')
    plt.hist(test_means, bins=30, alpha=0.7, label='Test', color='red')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    plt.title('Sample Mean Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Sample std distributions
    plt.subplot(2, 3, 3)
    train_stds = train_flat.std(dim=1)
    test_stds = test_flat.std(dim=1)
    plt.hist(train_stds, bins=30, alpha=0.7, label='Training', color='blue')
    plt.hist(test_stds, bins=30, alpha=0.7, label='Test', color='red')
    plt.xlabel('Sample Standard Deviation')
    plt.ylabel('Frequency')
    plt.title('Sample Std Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Class distributions
    plt.subplot(2, 3, 4)
    train_counts = Counter(train_labels.tolist())
    test_counts = Counter(test_labels.tolist())
    
    classes = sorted(train_counts.keys())
    train_class_counts = [train_counts[c] for c in classes]
    test_class_counts = [test_counts[c] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, train_class_counts, width, label='Training', alpha=0.7, color='blue')
    plt.bar(x + width/2, test_class_counts, width, label='Test', alpha=0.7, color='red')
    plt.xlabel('Digit Class')
    plt.ylabel('Count')
    plt.title('Class Distribution Comparison')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Mean images comparison
    plt.subplot(2, 3, 5)
    train_mean_img = train_images.mean(dim=0)
    train_mean_img = (train_mean_img + 1) / 2
    plt.imshow(train_mean_img.squeeze(), cmap='gray')
    plt.title('Training Set Mean Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    test_mean_img = test_images.mean(dim=0)
    test_mean_img = (test_mean_img + 1) / 2
    plt.imshow(test_mean_img.squeeze(), cmap='gray')
    plt.title('Test Set Mean Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/train_vs_test.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main visualization function."""
    print("=" * 60)
    print("MNIST Dataset Visualization")
    print("=" * 60)
    
    output_dir = "visualize_dataset/MNIST"

    # Load data
    train_dataset, test_dataset = load_mnist_data()
    
    # 1. Sample images
    visualize_sample_images(train_dataset, num_samples=16, title="MNIST Training Samples", output_dir=output_dir)
    visualize_sample_images(test_dataset, num_samples=16, title="MNIST Test Samples", output_dir=output_dir)
    
    # 2. Class distribution
    visualize_class_distribution(train_dataset, title="Training Set Class Distribution", output_dir=output_dir)
    visualize_class_distribution(test_dataset, title="Test Set Class Distribution", output_dir=output_dir)
    
    # 3. Image statistics
    visualize_image_statistics(train_dataset, num_samples=1000, output_dir=output_dir)
    
    # 4. Digit variations
    visualize_digit_variations(train_dataset, num_samples_per_digit=8, output_dir=output_dir)
    
    # 5. Training vs Test comparison
    visualize_training_vs_test(train_dataset, test_dataset, output_dir=output_dir  )
    
    print("\n" + "=" * 60)
    print("Visualization complete! Check the generated PNG files.")
    print("=" * 60)

if __name__ == "__main__":
    main()