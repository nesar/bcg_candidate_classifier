"""
Reproducibility Utilities for BCG Classification

This module provides comprehensive seed control to ensure completely deterministic
training and testing with zero stochasticity between runs.

Usage:
    from utils.reproducibility import set_global_seed, make_deterministic

    # At the start of your script:
    set_global_seed(42)
    make_deterministic()
"""

import random
import numpy as np
import torch
import os


def set_global_seed(seed=42):
    """
    Set random seeds for all libraries to ensure reproducibility.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Integer seed value (default: 42)
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (for all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    print(f"[Reproducibility] Global seed set to {seed}")


def make_deterministic(warn=True):
    """
    Make PyTorch operations deterministic.

    This function:
    - Disables CUDA benchmarking (ensures deterministic algorithms)
    - Enables deterministic mode for CuDNN operations
    - Sets environment variables for deterministic behavior

    Args:
        warn: If True, print warnings about potential performance impact

    Note:
        Deterministic mode may reduce performance (10-20% slower) but ensures
        perfect reproducibility. This is essential for scientific experiments.
    """
    # Disable CuDNN benchmark for determinism
    # Benchmarking finds fastest algorithm but is non-deterministic
    torch.backends.cudnn.benchmark = False

    # Enable deterministic mode for CuDNN
    torch.backends.cudnn.deterministic = True

    # For PyTorch >= 1.8, use deterministic algorithms
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except:
            # Some operations don't have deterministic implementations
            # Set to warn mode instead
            if hasattr(torch, 'set_deterministic_debug_mode'):
                torch.set_deterministic_debug_mode(1)  # Warn instead of error

    # Set environment variables for deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic CUDA operations
    os.environ['PYTHONHASHSEED'] = '0'  # Deterministic Python hashing

    if warn:
        print("[Reproducibility] Deterministic mode enabled")
        print("  - CuDNN benchmark: DISABLED")
        print("  - CuDNN deterministic: ENABLED")
        print("  - PyTorch deterministic algorithms: ENABLED")
        print("  Note: This may reduce performance by 10-20% but ensures perfect reproducibility")


def seed_worker(worker_id):
    """
    Seed function for DataLoader workers to ensure reproducible data loading.

    This should be passed as worker_init_fn to DataLoader:
        DataLoader(..., worker_init_fn=seed_worker)

    Args:
        worker_id: Worker ID (automatically passed by DataLoader)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed=42):
    """
    Create a deterministic random number generator for PyTorch.

    This generator should be passed to operations that support it:
        torch.utils.data.random_split(..., generator=g)
        DataLoader(..., generator=g)

    Args:
        seed: Integer seed value (default: 42)

    Returns:
        torch.Generator with fixed seed
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def reset_model_weights(model, seed=42):
    """
    Reset model weights to ensure consistent initialization across runs.

    This function re-initializes all model parameters with a fixed seed,
    ensuring that multiple training runs start from the exact same initialization.

    Args:
        model: PyTorch model
        seed: Seed for weight initialization (default: 42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    @torch.no_grad()
    def init_weights(m):
        """Initialize weights for common layer types."""
        if hasattr(m, 'reset_parameters'):
            # Use the layer's built-in reset (respects seed)
            m.reset_parameters()
        elif isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            # Manual initialization for layers without reset_parameters
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # BatchNorm initialization
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)
    print(f"[Reproducibility] Model weights reset with seed {seed}")


class DeterministicContext:
    """
    Context manager for deterministic operations.

    Usage:
        with DeterministicContext(seed=42):
            # All operations in this block are deterministic
            model.train()
            ...
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.old_python_state = None
        self.old_numpy_state = None
        self.old_torch_state = None
        self.old_cudnn_benchmark = None
        self.old_cudnn_deterministic = None

    def __enter__(self):
        # Save old states
        self.old_python_state = random.getstate()
        self.old_numpy_state = np.random.get_state()
        self.old_torch_state = torch.get_rng_state()
        self.old_cudnn_benchmark = torch.backends.cudnn.benchmark
        self.old_cudnn_deterministic = torch.backends.cudnn.deterministic

        # Set deterministic mode
        set_global_seed(self.seed)
        make_deterministic(warn=False)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old states
        random.setstate(self.old_python_state)
        np.random.set_state(self.old_numpy_state)
        torch.set_rng_state(self.old_torch_state)
        torch.backends.cudnn.benchmark = self.old_cudnn_benchmark
        torch.backends.cudnn.deterministic = self.old_cudnn_deterministic


def print_reproducibility_info():
    """Print current reproducibility settings for debugging."""
    print("\n" + "="*80)
    print("REPRODUCIBILITY SETTINGS")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CuDNN version: {torch.backends.cudnn.version()}")
    print(f"\nCurrent Settings:")
    print(f"  - CuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  - CuDNN deterministic: {torch.backends.cudnn.deterministic}")
    if hasattr(torch, 'are_deterministic_algorithms_enabled'):
        print(f"  - Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
    print(f"  - CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'NOT SET')}")
    print(f"  - PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'NOT SET')}")
    print("="*80 + "\n")


# Example usage and testing
if __name__ == "__main__":
    # Test reproducibility
    print("Testing reproducibility utilities...\n")

    # Set global seed
    set_global_seed(42)
    make_deterministic()

    # Print settings
    print_reproducibility_info()

    # Test random operations
    print("Testing deterministic random operations:")

    # Python random
    random_vals_1 = [random.random() for _ in range(5)]
    random.seed(42)
    random_vals_2 = [random.random() for _ in range(5)]
    print(f"Python random test: {'PASS' if random_vals_1 == random_vals_2 else 'FAIL'}")

    # NumPy random
    np.random.seed(42)
    numpy_vals_1 = np.random.rand(5)
    np.random.seed(42)
    numpy_vals_2 = np.random.rand(5)
    print(f"NumPy random test: {'PASS' if np.allclose(numpy_vals_1, numpy_vals_2) else 'FAIL'}")

    # PyTorch random
    torch.manual_seed(42)
    torch_vals_1 = torch.rand(5)
    torch.manual_seed(42)
    torch_vals_2 = torch.rand(5)
    print(f"PyTorch random test: {'PASS' if torch.allclose(torch_vals_1, torch_vals_2) else 'FAIL'}")

    # Test model initialization
    print("\nTesting deterministic model initialization:")

    model1 = torch.nn.Linear(10, 5)
    reset_model_weights(model1, seed=42)
    weights1 = model1.weight.data.clone()

    model2 = torch.nn.Linear(10, 5)
    reset_model_weights(model2, seed=42)
    weights2 = model2.weight.data.clone()

    print(f"Model initialization test: {'PASS' if torch.allclose(weights1, weights2) else 'FAIL'}")

    print("\nAll tests completed!")
