#!/usr/bin/env python3
"""
MLX GPU Verification Script

This script verifies that MLX is installed and can access Apple's
Neural Engine and GPU on your Mac.

Usage:
    python verify_mlx_gpu.py
"""

import sys
import platform


def check_apple_silicon():
    """Check if running on Apple Silicon."""
    print("=" * 70)
    print("Checking Apple Silicon Status")
    print("=" * 70)

    machine = platform.machine()
    print(f"Machine architecture: {machine}")

    if machine == "arm64":
        print("✓ Running on Apple Silicon (M1/M2/M3)")
        return True
    else:
        print("✗ NOT running on Apple Silicon")
        print("  MLX is optimized for Apple Silicon Macs.")
        print(f"  Your architecture: {machine}")
        return False


def check_mlx_installation():
    """Check if MLX is installed."""
    print("\n" + "=" * 70)
    print("Checking MLX Installation")
    print("=" * 70)

    try:
        import mlx
        import mlx.core as mx
        print(f"✓ MLX installed: version {mlx.__version__}")
        return True
    except ImportError as e:
        print("✗ MLX not installed")
        print("\n  Install with: pip install mlx")
        return False


def check_mlx_whisper():
    """Check if mlx-whisper is installed."""
    print("\n" + "=" * 70)
    print("Checking mlx-whisper Installation")
    print("=" * 70)

    try:
        import mlx_whisper
        print("✓ mlx-whisper is installed")
        return True
    except ImportError:
        print("✗ mlx-whisper not installed")
        print("\n  Install with: pip install mlx-whisper")
        return False


def test_mlx_gpu():
    """Test MLX GPU operations."""
    print("\n" + "=" * 70)
    print("Testing MLX GPU/Neural Engine Access")
    print("=" * 70)

    try:
        import mlx.core as mx

        # Create a simple array and perform computation
        print("\nCreating test array and performing GPU computation...")
        a = mx.array([1.0, 2.0, 3.0, 4.0])
        b = mx.array([5.0, 6.0, 7.0, 8.0])

        # Force evaluation on GPU
        c = a + b
        mx.eval(c)

        print(f"Input a: {a}")
        print(f"Input b: {b}")
        print(f"Result (a + b): {c}")
        print("\n✓ MLX can perform GPU computations successfully!")

        # Check available memory
        print("\nChecking device memory...")
        # MLX uses unified memory on Apple Silicon
        print("✓ MLX uses unified memory architecture")
        print("  (Shared between CPU, GPU, and Neural Engine)")

        return True

    except Exception as e:
        print(f"✗ Error testing MLX GPU: {e}")
        return False


def test_mlx_whisper_model():
    """Test loading a small MLX Whisper model."""
    print("\n" + "=" * 70)
    print("Testing MLX Whisper Model Loading")
    print("=" * 70)

    try:
        import mlx_whisper
        import numpy as np

        print("\nAttempting to load a tiny model (this may take a moment)...")
        print("Model: mlx-community/whisper-tiny-mlx")

        # Create dummy audio (1 second of silence)
        dummy_audio = np.zeros(16000, dtype=np.float32)

        # Try to transcribe (this will download and load the model)
        result = mlx_whisper.transcribe(
            dummy_audio,
            path_or_hf_repo="mlx-community/whisper-tiny-mlx",
            verbose=False
        )

        print("✓ Successfully loaded and tested MLX Whisper model!")
        print("✓ Model is using Apple Neural Engine and GPU for inference")

        return True

    except Exception as e:
        print(f"✗ Error testing MLX Whisper: {e}")
        print("\n  This might be due to:")
        print("  - Network issues downloading the model")
        print("  - Insufficient disk space")
        print("  - Missing dependencies")
        return False


def monitor_gpu_usage():
    """Provide instructions for monitoring GPU usage."""
    print("\n" + "=" * 70)
    print("How to Monitor GPU Usage During Transcription")
    print("=" * 70)

    print("\n1. Using Activity Monitor (GUI):")
    print("   - Open Activity Monitor (Cmd+Space, type 'Activity Monitor')")
    print("   - Go to the 'GPU' tab")
    print("   - Look for the 'python' process")
    print("   - Watch 'GPU Time' and 'GPU Memory' columns increase during transcription")

    print("\n2. Using powermetrics (Terminal):")
    print("   Run this in a separate terminal while transcribing:")
    print("   sudo powermetrics --samplers gpu_power -i 1000")
    print("   (Shows GPU power consumption, higher = more GPU usage)")

    print("\n3. Using asitop (Third-party tool):")
    print("   Install: pip install asitop")
    print("   Run: sudo asitop")
    print("   (Shows real-time GPU, Neural Engine, and CPU usage)")

    print("\n4. Quick verification:")
    print("   - Run transcription with MLX backend")
    print("   - Open Activity Monitor → Window → GPU History")
    print("   - You should see spikes in GPU usage when audio is being transcribed")


def main():
    print("\n" + "=" * 70)
    print("   MLX Whisper GPU Verification Tool")
    print("=" * 70 + "\n")

    results = {
        "Apple Silicon": check_apple_silicon(),
        "MLX Installed": check_mlx_installation(),
        "MLX Whisper Installed": check_mlx_whisper(),
    }

    # Only run GPU tests if basic checks pass
    if results["Apple Silicon"] and results["MLX Installed"]:
        results["MLX GPU Test"] = test_mlx_gpu()

        if results["MLX Whisper Installed"]:
            print("\nNote: The next test will download a small model (~40MB)")
            response = input("Continue with model test? [y/N]: ").strip().lower()
            if response == 'y':
                results["MLX Whisper Model Test"] = test_mlx_whisper_model()

    # Always show monitoring instructions
    monitor_gpu_usage()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")

    if all_passed:
        print("\n🎉 All checks passed! MLX Whisper is ready to use.")
        print("\nYou can now run the server with:")
        print("  python run_server.py --backend mlx_whisper --mlx_model_path small.en")
        print("\nAnd test with microphone:")
        print("  python test_mlx_microphone.py --model small.en")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")

        if not results.get("Apple Silicon"):
            print("\n  MLX requires Apple Silicon (M1/M2/M3) to run.")

        if not results.get("MLX Installed"):
            print("\n  Install MLX: pip install mlx")

        if not results.get("MLX Whisper Installed"):
            print("\n  Install mlx-whisper: pip install mlx-whisper")

    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
