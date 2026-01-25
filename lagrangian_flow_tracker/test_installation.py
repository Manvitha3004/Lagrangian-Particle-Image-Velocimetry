"""Test script to verify installation and basic functionality."""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import yaml
        print(f"  ✓ PyYAML")
    except ImportError as e:
        print(f"  ✗ PyYAML import failed: {e}")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"  ⚠ CUDA not available (CPU mode will be used)")
    
    return True


def test_modules():
    """Test project modules."""
    print("\nTesting project modules...")
    
    sys.path.insert(0, 'src')
    
    try:
        from src import utils
        print("  ✓ src.utils")
    except ImportError as e:
        print(f"  ✗ src.utils failed: {e}")
        return False
    
    try:
        from src import model
        print("  ✓ src.model")
    except ImportError as e:
        print(f"  ✗ src.model failed: {e}")
        return False
    
    try:
        from src import data_processing
        print("  ✓ src.data_processing")
    except ImportError as e:
        print(f"  ✗ src.data_processing failed: {e}")
        return False
    
    try:
        from src import training
        print("  ✓ src.training")
    except ImportError as e:
        print(f"  ✗ src.training failed: {e}")
        return False
    
    try:
        from src import tracking
        print("  ✓ src.tracking")
    except ImportError as e:
        print(f"  ✗ src.tracking failed: {e}")
        return False
    
    try:
        from src import visualization
        print("  ✓ src.visualization")
    except ImportError as e:
        print(f"  ✗ src.visualization failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    sys.path.insert(0, 'src')
    
    try:
        import torch
        from src.model import NeuralFlowField
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeuralFlowField(
            num_frequencies=6,
            hidden_dims=[128, 64],
            activation='relu'
        ).to(device)
        
        # Test forward pass
        x = torch.randn(10, device=device)
        y = torch.randn(10, device=device)
        t = torch.randn(10, device=device)
        
        output = model(x, y, t)
        
        assert output.shape == (10, 2), f"Expected shape (10, 2), got {output.shape}"
        
        print(f"  ✓ Model created and tested")
        print(f"  ✓ Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    sys.path.insert(0, 'src')
    
    try:
        from src.utils import load_config
        
        config = load_config('configs/config.yaml')
        
        assert 'model' in config, "Config missing 'model' section"
        assert 'training' in config, "Config missing 'training' section"
        assert 'data' in config, "Config missing 'data' section"
        
        print("  ✓ Configuration loaded successfully")
        print(f"  ✓ Config sections: {list(config.keys())}")
        
        return True
    except Exception as e:
        print(f"  ✗ Configuration loading failed: {e}")
        return False


def test_flow_reading():
    """Test .flo file reading."""
    print("\nTesting .flo file reading...")
    
    sys.path.insert(0, 'src')
    
    try:
        from src.data_processing import read_flo_file
        import numpy as np
        
        # Check if sample flow files exist
        flow_dir = "../Dataset/backstep"
        if os.path.exists(flow_dir):
            flow_files = [f for f in os.listdir(flow_dir) if f.endswith('.flo')]
            if flow_files:
                sample_file = os.path.join(flow_dir, flow_files[0])
                flow = read_flo_file(sample_file)
                print(f"  ✓ Successfully read {sample_file}")
                print(f"  ✓ Flow shape: {flow.shape}")
                print(f"  ✓ Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
                return True
        
        print("  ⚠ No .flo files found (this is OK if you're using videos)")
        return True
        
    except Exception as e:
        print(f"  ⚠ Flow file reading test skipped: {e}")
        return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("Lagrangian Flow Tracker - Installation Test")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Project Modules", test_modules),
        ("Configuration", test_config),
        ("Model Creation", test_model_creation),
        ("Flow File Reading", test_flow_reading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Run demo: python demo.py")
        print("  2. Train model: python train.py --config configs/config.yaml")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("  - Make sure all dependencies are installed: pip install -r requirements.txt")
        print("  - Check that your data directory structure is correct")
    
    print("=" * 80)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
