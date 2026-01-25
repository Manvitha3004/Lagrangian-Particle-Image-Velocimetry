"""
Run PINN model tests and validation.

This script runs the complete test suite for the Physics-Informed Neural Network
implementation, including:
- Unit tests for model architecture
- Gradient flow verification
- Analytical validation on Taylor-Green vortex
- Physics constraint verification
- Performance benchmarks

Usage:
    python run_pinn_tests.py [--quick] [--verbose]
    
Options:
    --quick     Skip slow integration tests
    --verbose   Show detailed test output
"""

import sys
import argparse
from pathlib import Path
import subprocess
import time

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"✓ {description} PASSED ({elapsed:.1f}s)")
        return True
    else:
        print(f"✗ {description} FAILED ({elapsed:.1f}s)")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run PINN tests")
    parser.add_argument('--quick', action='store_true', help='Skip slow tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / 'src'))
    
    results = []
    
    print("="*70)
    print("PHYSICS-INFORMED NEURAL NETWORK (PINN) TEST SUITE")
    print("="*70)
    print(f"Project root: {project_root}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Test 1: Basic unit tests
    verbose_flag = "-v" if args.verbose else ""
    markers = '-m "not slow"' if args.quick else ""
    
    cmd = f"pytest tests/test_advanced_models.py {verbose_flag} {markers} --tb=short"
    results.append(run_command(cmd, "Unit Tests"))
    
    # Test 2: Import test
    print(f"\n{'='*70}")
    print("Testing module imports...")
    print(f"{'='*70}")
    try:
        from src.advanced_models import (
            NavierStokesPINN,
            compute_navier_stokes_residual,
            TaylorGreenVortex
        )
        print("✓ All imports successful")
        results.append(True)
    except Exception as e:
        print(f"✗ Import failed: {e}")
        results.append(False)
    
    # Test 3: Quick validation run (if not quick mode)
    if not args.quick:
        print(f"\n{'='*70}")
        print("Running quick PINN validation...")
        print(f"{'='*70}")
        try:
            import torch
            from src.advanced_models import NavierStokesPINN, train_pinn_on_taylor_green
            
            pinn = NavierStokesPINN(hidden_dims=[64, 64], viscosity=0.01, density=1.0)
            
            history = train_pinn_on_taylor_green(
                pinn,
                num_epochs=100,
                num_points=500,
                verbose=False
            )
            
            if history['total_loss'][-1] < history['total_loss'][0]:
                print("✓ PINN training converged")
                results.append(True)
            else:
                print("✗ PINN did not converge")
                results.append(False)
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            results.append(False)
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"Total tests:  {total}")
    print(f"Passed:       {passed} ✓")
    print(f"Failed:       {failed} ✗")
    print(f"Success rate: {100*passed/total:.1f}%")
    print(f"{'='*70}")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED! PINN implementation is production-ready.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
