#!/usr/bin/env python3

import os, sys, time, platform, shutil, statistics, subprocess
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import psutil

# Optional libs
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class MatrixBenchmark:
    def __init__(self, iterations: int = 30):
        self.iter = iterations
        self.results: Dict[str, Dict] = {}
        self.matrix_sizes: List[int] = []

    @staticmethod
    def gflops(n: int, dt: float) -> float:
        ops = 2 * (n**3) - (n**2)
        return ops / (dt * 1e9)

    @staticmethod
    def gen(n: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        return np.random.rand(n, n).astype(dtype), np.random.rand(n, n).astype(dtype)

    # ---------------- Task 1: naive Python ----------------
    @staticmethod
    def naive_py(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        C = np.zeros((n, n), dtype=A.dtype)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += float(A[i, k]) * float(B[k, j])
                C[i, j] = s
        return C

    def bench_naive_py(self, sizes: List[int]) -> Dict:
        print("\n🐍 TASK 1: Naive Python")
        out = {'sizes': [], 'mean_gflops': [], 'std_gflops': [], 'times': []}
        for n in sizes:
            print(f"  n={n}")
            A, B = self.gen(n)
            g_list, t_list = [], []
            for _ in range(self.iter):
                t0 = time.perf_counter(); _ = self.naive_py(A, B); t1 = time.perf_counter()
                dt = t1 - t0
                g_list.append(self.gflops(n, dt)); t_list.append(dt)
            m, s = statistics.mean(g_list), (statistics.stdev(g_list) if len(g_list) > 1 else 0.0)
            out['sizes'].append(n); out['mean_gflops'].append(m); out['std_gflops'].append(s); out['times'].append(statistics.mean(t_list))
            print(f"    {m:.4f} ± {s:.4f} GFLOPS, mean time {statistics.mean(t_list):.4f}s")
        self.results['naive_python'] = out
        return out

    # ---------------- Task 2: NumPy (CPU) ----------------
    def bench_numpy(self, sizes: List[int]) -> Dict:
        print("\nTASK 2: NumPy (CPU)")
        out = {'sizes': [], 'mean_gflops': [], 'std_gflops': [], 'times': []}
        for n in sizes:
            print(f"  n={n}")
            A, B = self.gen(n)
            g_list, t_list = [], []
            for _ in range(self.iter):
                t0 = time.perf_counter(); _ = A @ B; t1 = time.perf_counter()
                dt = t1 - t0
                g_list.append(self.gflops(n, dt)); t_list.append(dt)
            m, s = statistics.mean(g_list), (statistics.stdev(g_list) if len(g_list) > 1 else 0.0)
            out['sizes'].append(n); out['mean_gflops'].append(m); out['std_gflops'].append(s); out['times'].append(statistics.mean(t_list))
            print(f"    {m:.4f} ± {s:.4f} GFLOPS, mean time {statistics.mean(t_list):.6f}s")
        self.results['numpy'] = out
        return out

    # ---------------- Task 3: CuPy (GPU) ----------------
    def bench_cupy_fp32(self, sizes: List[int]) -> Dict:
        if not CUPY_AVAILABLE:
            print("\n🎮 TASK 3a: CuPy FP32 not available (pip install cupy-cuda12x)")
            return {}
        print("\n🎮 TASK 3a: CuPy FP32 (GPU)")
        out = {'sizes': [], 'mean_gflops': [], 'std_gflops': [], 'times': []}
        for n in sizes:
            print(f"  n={n}")
            A, B = self.gen(n, np.float32)
            Ag, Bg = cp.asarray(A), cp.asarray(B)
            for _ in range(3): _ = Ag @ Bg; cp.cuda.Stream.null.synchronize()
            g_list, t_list = [], []
            for _ in range(self.iter):
                t0 = time.perf_counter(); _ = Ag @ Bg; cp.cuda.Stream.null.synchronize(); t1 = time.perf_counter()
                dt = t1 - t0
                g_list.append(self.gflops(n, dt)); t_list.append(dt)
            m, s = statistics.mean(g_list), (statistics.stdev(g_list) if len(g_list) > 1 else 0.0)
            out['sizes'].append(n); out['mean_gflops'].append(m); out['std_gflops'].append(s); out['times'].append(statistics.mean(t_list))
            print(f"    {m:.4f} ± {s:.4f} GFLOPS, mean time {statistics.mean(t_list):.6f}s")
        self.results['cupy_fp32'] = out
        return out

    def bench_cupy_fp64(self, sizes: List[int]) -> Dict:
        if not CUPY_AVAILABLE:
            print("\nTASK 3b: CuPy FP64 not available (pip install cupy-cuda12x)")
            return {}
        print("\nTASK 3b: CuPy FP64 (GPU) — slow on consumer GPUs")
        out = {'sizes': [], 'mean_gflops': [], 'std_gflops': [], 'times': []}
        for n in sizes:
            print(f"  n={n}")
            A, B = self.gen(n, np.float64)
            Ag, Bg = cp.asarray(A), cp.asarray(B)
            for _ in range(3): _ = Ag @ Bg; cp.cuda.Stream.null.synchronize()
            g_list, t_list = [], []
            for _ in range(self.iter):
                t0 = time.perf_counter(); _ = Ag @ Bg; cp.cuda.Stream.null.synchronize(); t1 = time.perf_counter()
                dt = t1 - t0
                g_list.append(self.gflops(n, dt)); t_list.append(dt)
            m, s = statistics.mean(g_list), (statistics.stdev(g_list) if len(g_list) > 1 else 0.0)
            out['sizes'].append(n); out['mean_gflops'].append(m); out['std_gflops'].append(s); out['times'].append(statistics.mean(t_list))
            print(f"    {m:.4f} ± {s:.4f} GFLOPS, mean time {statistics.mean(t_list):.6f}s")
        self.results['cupy_fp64'] = out
        return out

    # ---------------- Task 4: Naive C via ctypes ----------------
    @staticmethod
    def _gcc_target() -> str:
        try:
            out = subprocess.check_output(["gcc", "-v"], stderr=subprocess.STDOUT, text=True)
            for line in out.splitlines():
                if "Target:" in line:
                    return line.split("Target:", 1)[1].strip()
        except Exception:
            return "Unknown"
        return "Unknown"

    def _compile_c(self) -> str:
        c_src = "matrix_mult.c"
        if not os.path.exists(c_src):
            print(f" {c_src} not found next to this script.")
            return ""
        if shutil.which("gcc") is None:
            print(" gcc not found on PATH. Install MSYS2/MinGW-w64 (x86_64).")
            return ""
        is_win = platform.system() == "Windows"
        lib_name = "matrix_mult.dll" if is_win else "matrix_mult.so"
        cc = os.environ.get("CC", "gcc")

        # Warn if 32-bit GCC with 64-bit Python
        import struct
        py_bits = struct.calcsize("P") * 8
        tgt = self._gcc_target()
        if is_win and py_bits == 64 and "i686" in tgt:
            print(f"  32-bit GCC detected (Target: {tgt}) with 64-bit Python. DLL will fail to load.")
            print("    Install a 64-bit GCC (x86_64-w64-mingw32), or set CC to it.")
            return ""

        if is_win:
            cmd = f'"{cc}" -shared -o {lib_name} {c_src}'
        else:
            cmd = f'"{cc}" -shared -fPIC -o {lib_name} {c_src}'
        print(f"  Compiling C library: {cmd}")
        rc = os.system(cmd)
        if rc != 0 or not os.path.exists(lib_name):
            print(" Failed to compile C library.")
            return ""
        return lib_name

    def bench_c(self, sizes: List[int]) -> Dict:
        print("\n  TASK 4: Naive C (ctypes)")
        lib_path = self._compile_c()
        if not lib_path:
            print("   Skipping C benchmark.")
            return {}

        import ctypes
        try:
            lib = ctypes.CDLL(os.path.abspath(lib_path))
        except OSError as e:
            print(" Failed to load C DLL/SO:", e)
            print("   Skipping C benchmark.")
            return {}

        lib.c_matrix_multiply.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        lib.c_matrix_multiply.restype = None

        out = {'sizes': [], 'mean_gflops': [], 'std_gflops': [], 'times': []}
        for n in sizes:
            print(f"  n={n}")
            A, B = self.gen(n, np.float32)
            C = np.zeros((n, n), dtype=np.float32)
            A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            g_list, t_list = [], []
            for _ in range(self.iter):
                t0 = time.perf_counter(); lib.c_matrix_multiply(A_ptr, B_ptr, C_ptr, n); t1 = time.perf_counter()
                dt = t1 - t0
                g_list.append(self.gflops(n, dt)); t_list.append(dt)
            m, s = statistics.mean(g_list), (statistics.stdev(g_list) if len(g_list) > 1 else 0.0)
            out['sizes'].append(n); out['mean_gflops'].append(m); out['std_gflops'].append(s); out['times'].append(statistics.mean(t_list))
            print(f"    {m:.4f} ± {s:.4f} GFLOPS, mean time {statistics.mean(t_list):.4f}s")
        self.results['c_implementation'] = out
        return out

    # ---------------- Plots ----------------
    def save_method_plots(self):
        labels = {
            'naive_python': 'Python Loop Performance',
            'numpy': 'NumPy Performance (CPU)',
            'cupy_fp32': 'CuPy Single Precision (FP32)',
            'cupy_fp64': 'CuPy Double Precision (FP64)',
            'c_implementation': 'C Loop Performance'
        }
        for key, title in labels.items():
            data = self.results.get(key)
            if not data:
                continue
            plt.figure(figsize=(8, 6))
            plt.errorbar(
                data['sizes'], data['mean_gflops'],
                yerr=data['std_gflops'], marker='o', linestyle='-',
                capsize=5, linewidth=2, markersize=6
            )
            plt.xlabel('Matrix Size (n)'); plt.ylabel('Performance (Mean GFLOPS)')
            plt.title(title); plt.grid(True, alpha=0.3)
            plt.savefig(f'{key}_performance.png', dpi=300, bbox_inches='tight')
            plt.close()

    def save_comparison_plot(self):
        plt.figure(figsize=(10, 7))
        order = ['naive_python', 'numpy', 'cupy_fp32', 'cupy_fp64', 'c_implementation']
        names = {
            'naive_python': 'Python Loops',
            'numpy': 'NumPy (CPU)',
            'cupy_fp32': 'CuPy FP32 (GPU)',
            'cupy_fp64': 'CuPy FP64 (GPU)',
            'c_implementation': 'C Loops'
        }
        for k in order:
            d = self.results.get(k)
            if not d: continue
            plt.errorbar(
                d['sizes'], d['mean_gflops'],
                yerr=d['std_gflops'], marker='o', linestyle='-',
                capsize=5, linewidth=2, markersize=6, label=names[k]
            )
        plt.xlabel('Matrix Size (n)'); plt.ylabel('Performance (Mean GFLOPS)')
        plt.title('Matrix Multiplication Performance Comparison')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------- System info ----------------
    def print_system_info(self):
        print("\n" + "="*60)
        print("SYSTEM SPECIFICATIONS FOR REPORT")
        print("="*60)
        try:
            import cpuinfo
            cpu = cpuinfo.get_cpu_info().get('brand_raw', platform.processor())
        except Exception:
            cpu = platform.processor() or "Unknown CPU"
        print(f"    CPU: {cpu}")
        print(f"   Architecture: {platform.machine()}")
        print(f"   Cores: {psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count(logical=True)} Logical")
        if CUPY_AVAILABLE:
            try:
                p = cp.cuda.runtime.getDeviceProperties(0)
                name = p['name'].decode()
                mem = p['totalGlobalMem']/(1024**3)
                print(f"GPU: {name} ({mem:.1f} GB)")
            except Exception:
                print("GPU: CUDA device detected")
        else:
            print("GPU: Not available/detected")
        print(f"Python: {sys.version.split()[0]}")
        print(f"NumPy: {np.__version__}")
        if CUPY_AVAILABLE: print(f"🎮 CuPy: {cp.__version__}")
        if TORCH_AVAILABLE: print(f"🔥 PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

    # ---------------- Orchestration ----------------
    def run_all(self, sizes: List[int]):
        self.matrix_sizes = sizes
        print("🚀 STARTING COMPLETE MATRIX MULTIPLICATION BENCHMARK")
        print(f"📏 Matrix sizes: {sizes}")
        print(f"🔄 Iterations per size: {self.iter}")

        self.print_system_info()

        self.bench_naive_py(sizes)
        self.bench_numpy(sizes)
        if CUPY_AVAILABLE:
            self.bench_cupy_fp32(sizes)
            self.bench_cupy_fp64(sizes)
        else:
            print("⚠️  Skipping CuPy tasks (install with: pip install cupy-cuda12x)")
        self.bench_c(sizes)

        self.save_method_plots()
        self.save_comparison_plot()
        self.summary()

    def summary(self):
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        names = {
            'naive_python': 'Naive Python',
            'numpy': 'NumPy (Optimized CPU)',
            'cupy_fp32': 'CuPy FP32 (GPU)',
            'cupy_fp64': 'CuPy FP64 (GPU)',
            'c_implementation': 'Naive C'
        }
        for k, d in self.results.items():
            if not d: continue
            peak = max(d['mean_gflops']); idx = d['mean_gflops'].index(peak); n_at = d['sizes'][idx]
            print(f"\n📊 {names.get(k,k)}")
            print("-"*40)
            print(f"   Peak: {peak:.2f} GFLOPS @ n={n_at}")
            print(f"   Sizes: {d['sizes']}")
            print(f"   Range: {min(d['mean_gflops']):.2f} – {peak:.2f} GFLOPS")


def main():
    print("🚀 HW3: Matrix Multiplication Benchmarks")
    sizes = [64, 128, 192, 256, 320]  # 5 common sizes across all tasks
    bench = MatrixBenchmark(iterations=30)
    bench.run_all(sizes)


if __name__ == "__main__":
    main()
