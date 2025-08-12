# Hardware Detection Script
# Run this to get your complete system specifications

import platform
import subprocess
import psutil
import GPUtil
import cpuinfo
import numpy as np

def get_system_info():
    print("="*50)
    print("SYSTEM HARDWARE SPECIFICATIONS")
    print("="*50)
    
    # CPU Information
    print("\n🔧 CPU DETAILS:")
    print("-"*30)
    cpu_info = cpuinfo.get_cpu_info()
    print(f"CPU Brand: {cpu_info.get('brand_raw', 'Unknown')}")
    print(f"Architecture: {cpu_info.get('arch', 'Unknown')}")
    print(f"Cores: {psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count(logical=True)} Logical")
    print(f"Base Frequency: {cpu_info.get('hz_advertised_friendly', 'Unknown')}")
    print(f"Max Frequency: {psutil.cpu_freq().max if psutil.cpu_freq() else 'Unknown'} MHz")
    
    # For ASUS A15, likely AMD Ryzen 7 4800H or similar
    # Theoretical Peak FLOPS calculation for common ASUS A15 CPUs:
    if "AMD" in cpu_info.get('brand_raw', ''):
        print(f"Platform: {platform.system()} {platform.release()}")
        print("\n📊 ESTIMATED THEORETICAL PEAK PERFORMANCE:")
        cores = psutil.cpu_count(logical=False)
        # AMD Zen 2/3 can do 16 FP32 ops per cycle with AVX2
        base_freq = 2.9  # Typical for Ryzen 7 4800H (adjust based on your model)
        theoretical_gflops = cores * base_freq * 16  # 16 FP32 ops per cycle
        print(f"CPU Theoretical Peak (FP32): ~{theoretical_gflops:.1f} GFLOPS")
    
    # Memory Information
    print(f"\n💾 MEMORY:")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU Information
    print(f"\n🎮 GPU DETAILS:")
    print("-"*30)
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
                print(f"Memory: {gpu.memoryTotal} MB")
                print(f"Driver: {gpu.driver}")
                
                # Common ASUS A15 GPUs and their specs:
                gpu_specs = {
                    "GTX 1650": {"fp32_tflops": 2.9, "memory": 4},
                    "GTX 1660 Ti": {"fp32_tflops": 5.4, "memory": 6},
                    "RTX 3050": {"fp32_tflops": 9.1, "memory": 4},
                    "RTX 3060": {"fp32_tflops": 13.2, "memory": 6},
                }
                
                for gpu_name, specs in gpu_specs.items():
                    if gpu_name.replace(" ", "").lower() in gpu.name.replace(" ", "").lower():
                        print(f"Theoretical Peak (FP32): ~{specs['fp32_tflops']} TFLOPS")
                        break
        else:
            print("No dedicated GPU detected")
            print("Integrated graphics detected (use CPU for benchmarking)")
    except Exception as e:
        print(f"Could not detect GPU: {e}")
        print("You may have integrated graphics only")
    
    # Check CUDA availability
    print(f"\n🔥 CUDA SUPPORT:")
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
    except:
        print("PyTorch not installed")
    
    try:
        import cupy
        print(f"CuPy available: True")
    except:
        print("CuPy not installed")

def install_requirements():
    """Install required packages"""
    packages = [
        "numpy", "cupy-cuda11x", "matplotlib", "psutil", 
        "GPUtil", "py-cpuinfo", "torch"
    ]
    
    print("\n📦 INSTALLATION COMMANDS:")
    print("Run these commands to install required packages:")
    for package in packages:
        print(f"pip install {package}")

if __name__ == "__main__":
    try:
        get_system_info()
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install required packages first:")
        install_requirements()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("- Use Google Colab T4 GPU for GPU benchmarks (free)")
    print("- Use your laptop CPU for CPU benchmarks")
    print("- T4 GPU specs: ~8.1 TFLOPS FP32, 16GB memory")