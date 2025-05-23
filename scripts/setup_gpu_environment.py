import subprocess
import sys
import platform


def run_command(command, description=""):
    """Run a shell command and return success status."""
    print(f"🔧 {description}")
    print(f"   Command: {command}")

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success!")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed!")
        if e.stderr.strip():
            print(f"   Error: {e.stderr.strip()}")
        return False


def detect_cuda_version():
    """Detect available CUDA version."""
    print("🔍 Detecting CUDA version...")

    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            if 'release 11.8' in output:
                return "11.8", "cu118"
            elif 'release 12.1' in output:
                return "12.1", "cu121"
            elif 'release 12.0' in output:
                return "12.0", "cu121"  # Use cu121 for 12.0
            elif 'release 11.' in output:
                return "11.x", "cu118"  # Default to 11.8 for other 11.x
            elif 'release 12.' in output:
                return "12.x", "cu121"  # Default to 12.1 for other 12.x
            else:
                print("   Unknown CUDA version, defaulting to 11.8")
                return "11.8", "cu118"
        else:
            print("   nvcc not found, defaulting to CUDA 11.8")
            return "11.8", "cu118"
    except:
        print("   Error detecting CUDA, defaulting to CUDA 11.8")
        return "11.8", "cu118"


def setup_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    print("\n🎮 Setting up PyTorch with CUDA support...")

    # Detect CUDA version
    cuda_version, pytorch_cuda = detect_cuda_version()
    print(f"   Detected CUDA: {cuda_version}")
    print(f"   Using PyTorch index: {pytorch_cuda}")

    # Uninstall existing PyTorch
    print("\n1. Removing existing PyTorch installations...")
    uninstall_cmd = f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio"
    run_command(uninstall_cmd, "Uninstalling existing PyTorch")

    # Install CUDA-enabled PyTorch
    print(f"\n2. Installing PyTorch with CUDA {cuda_version} support...")

    if pytorch_cuda == "cu118":
        install_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:  # cu121
        install_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

    success = run_command(install_cmd, f"Installing PyTorch with CUDA {cuda_version}")

    if success:
        print("✅ PyTorch installation completed!")
    else:
        print("❌ PyTorch installation failed!")
        print("💡 Try manual installation:")
        print(f"   {install_cmd}")

    return success


def install_remaining_requirements():
    """Install the remaining requirements."""
    print("\n📦 Installing remaining requirements...")

    # Install other GPU-optimized packages
    remaining_packages = [
        "sentence-transformers==2.2.2",
        "transformers==4.35.0",
        "accelerate==0.24.0",
        "qdrant-client==1.7.0",
        "numpy==1.24.0",
        "python-dotenv==1.0.0"
    ]

    for package in remaining_packages:
        cmd = f"{sys.executable} -m pip install {package}"
        run_command(cmd, f"Installing {package}")


def test_gpu_setup():
    """Test if GPU setup is working."""
    print("\n🧪 Testing GPU setup...")

    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")

        if cuda_available:
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            # Test GPU computation
            print("\n   Testing GPU computation...")
            device = torch.device('cuda')
            test_tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(test_tensor, test_tensor)
            print("   ✅ GPU computation test passed!")

            return True
        else:
            print("   ❌ CUDA not available")
            return False

    except ImportError:
        print("   ❌ PyTorch import failed")
        return False
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False


def test_sentence_transformers():
    """Test sentence transformers GPU support."""
    print("\n🧠 Testing Sentence Transformers...")

    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")

        # Test loading a small model
        print("   Loading test model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Check device
        import torch
        if torch.cuda.is_available():
            model = model.to('cuda')
            print(f"   Model device: {model.device}")

            # Test encoding
            test_text = "This is a test sentence for GPU processing."
            embedding = model.encode([test_text])
            print(f"   ✅ Test encoding successful! Shape: {embedding.shape}")

            return True
        else:
            print("   ⚠️  CUDA not available, model will use CPU")
            return False

    except Exception as e:
        print(f"   ❌ Sentence Transformers test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 GPU Environment Setup for RTX 4080")
    print("=" * 50)

    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"Virtual Environment: Active")
    else:
        print("⚠️  Not in a virtual environment - consider using one")

    # Setup steps
    steps = [
        ("Setting up PyTorch with CUDA", setup_pytorch_cuda),
        ("Installing remaining requirements", install_remaining_requirements),
        ("Testing GPU setup", test_gpu_setup),
        ("Testing Sentence Transformers", test_sentence_transformers)
    ]

    results = []

    for step_name, step_func in steps:
        print(f"\n{'=' * 20} {step_name} {'=' * 20}")
        try:
            success = step_func()
            results.append((step_name, success))
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            results.append((step_name, False))

    # Summary
    print(f"\n{'=' * 50}")
    print("📊 SETUP SUMMARY")
    print(f"{'=' * 50}")

    all_good = True
    for step_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
        if not success:
            all_good = False

    if all_good:
        print(f"\n🎉 GPU setup completed successfully!")
        print(f"✅ Your RTX 4080 is ready for high-performance embedding generation!")
        print(f"\nNext steps:")
        print(f"1. Run: python gpu_diagnostic.py  (to double-check)")
        print(f"2. Run: python preprocess_documents.py")
        print(f"3. Run: python index_to_qdrant.py")
    else:
        print(f"\n🔧 Some issues were detected.")
        print(f"💡 Manual troubleshooting:")
        print(f"1. Check NVIDIA drivers are installed")
        print(f"2. Install CUDA toolkit from nvidia.com")
        print(f"3. Try manual PyTorch installation:")
        print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


if __name__ == "__main__":
    main()