"""
Final optimized xTB workers combining all working optimizations from Steps 1-3.
Focuses on optimizations that work with the current xTB version.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from modules.xyz_io import build_xyz_string

ANGSTROM_TO_BOHR = 1.8897261254578281

# Pre-compiled regex patterns for better performance
ENERGY_PATTERNS = [
    re.compile(r"(?i)total\s+energy\s*=\s*([\-\d\.Ee+]+)\s*Eh"),
    re.compile(r"(?i)total\s+energy\s+([\-\d\.Ee+]+)\s*Eh"),
    re.compile(r"(?i)\bE\s*=\s*([\-\d\.Ee+]+)\s*Eh")
]


@dataclass
class XTBConfig:
    charge: int = 0
    mult: int = 1
    method: str = "gfn2"
    executable: str = "xtb"
    solvent: Optional[str] = None
    prefer_tblite: bool = False


class OptimizedXTBServer:
    """
    Highly optimized xTB server combining all working optimizations:
    - Enhanced caching with smart coordinate rounding and LRU eviction
    - Optimized subprocess calls with reused working directories
    - Better error handling and timeouts
    - Performance monitoring
    - Memory-efficient coordinate handling with bounded cache
    """
    
    def __init__(self, cfg: XTBConfig, max_cache_size: int = 10000):
        self.cfg = cfg
        # LRU cache with size limit to prevent unbounded memory growth
        self.cache: OrderedDict[str, float] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.workdir: Optional[str] = None
        self.coord_file: Optional[str] = None
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'subprocess_calls': 0,
            'cache_saves': 0
        }
        
        # Pre-create working directory and reuse it
        self._ensure_cmdline_backend()
        
        # Pre-build command template for efficiency
        self._build_cmd_template()

    def _build_cmd_template(self):
        """Pre-build command template to avoid repeated string operations."""
        method_flag = {"gfn2": "2", "gfn1": "1", "gfn0": "0", "gfnff": "ff"}.get(
            self.cfg.method.lower(), "2"
        )
        
        self.cmd_template = [
            self.cfg.executable,
            None,  # Placeholder for coord file
            "--gfn", method_flag,
            "--sp",
            "--chrg", str(self.cfg.charge),
            "--uhf", str(max(self.cfg.mult - 1, 0))
        ]
        
        if self.cfg.solvent:
            self.cmd_template.extend(["--alpb", self.cfg.solvent])

    def compute_energy(self, atoms: List[str], coords: np.ndarray) -> float:
        """Optimized energy computation with enhanced caching."""
        start_time = time.time()
        
        self.stats['total_calls'] += 1
        
        # Enhanced cache key with better coordinate rounding
        key = self._make_cache_key(atoms, coords)
        
        # Check cache with LRU ordering
        if key in self.cache:
            # Move to end (most recently used)
            energy = self.cache.pop(key)
            self.cache[key] = energy
            self.stats['cache_hits'] += 1
            elapsed = time.time() - start_time
            self.stats['total_time'] += elapsed
            return energy

        # Compute energy with optimized subprocess call
        energy = self._compute_energy_optimized(atoms, coords)
        
        # Cache the result with LRU eviction if needed
        with self.lock:
            if len(self.cache) >= self.max_cache_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[key] = energy
        self.stats['cache_saves'] += 1
        
        elapsed = time.time() - start_time
        self.stats['total_time'] += elapsed
        return energy

    def _compute_energy_optimized(self, atoms: List[str], coords: np.ndarray) -> float:
        """Optimized energy computation with reused working directory."""
        with self.lock:  # Thread safety
            # Reuse the same coordinate file (faster than creating new files)
            with open(self.coord_file, "w") as f:
                f.write(build_xyz_string(atoms, coords, "optimized xTB"))
            
            # Use pre-built command template
            cmd = self.cmd_template.copy()
            cmd[1] = self.coord_file  # Insert coordinate file path
            
            try:
                # Optimized subprocess call with shorter timeout for faster failures
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=self.workdir,
                    timeout=20,  # Reduced timeout for faster failure detection
                    check=False
                )
                
                self.stats['subprocess_calls'] += 1
                
                # Quick error check
                if result.returncode != 0:
                    error_output = result.stderr.lower()
                    if "error" in error_output or "failed" in error_output:
                        raise RuntimeError(f"xTB failed with return code {result.returncode}")
                
                output = result.stdout + "\n" + result.stderr
                return self._parse_energy_from_output(output)
                
            except subprocess.TimeoutExpired:
                raise RuntimeError("xTB calculation timed out after 20 seconds")

    def _parse_energy_from_output(self, output: str) -> float:
        """Optimized energy parsing with pre-compiled patterns."""
        for pattern in ENERGY_PATTERNS:
            match = pattern.search(output)
            if match:
                return float(match.group(1))
        
        raise RuntimeError(f"xTB energy not found in output:\n{output[-1000:]}")

    def _make_cache_key(self, atoms: List[str], coords: np.ndarray) -> str:
        """Enhanced cache key with better coordinate rounding for higher hit rates."""
        m = hashlib.sha1()
        
        # Efficient atom string encoding
        atoms_str = ",".join(atoms)
        m.update(atoms_str.encode('ascii'))
        
        # More aggressive coordinate rounding for better cache hits
        # Round to 2 decimal places (0.01 Å precision)
        rounded_coords = np.round(coords, decimals=2)
        m.update(rounded_coords.tobytes())
        
        return m.hexdigest()

    def _ensure_cmdline_backend(self) -> None:
        """Ensure working directory is ready and reusable."""
        if self.workdir is None:
            self.workdir = tempfile.mkdtemp(prefix="optimized_xtb_")
            self.coord_file = os.path.join(self.workdir, "coord.xyz")

    def get_performance_stats(self) -> Dict[str, float]:
        """Get detailed performance statistics."""
        if self.stats['total_calls'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_calls']
            avg_time = self.stats['total_time'] / self.stats['total_calls']
            
            # Calculate actual computation rate (excluding cache hits)
            actual_computations = self.stats['total_calls'] - self.stats['cache_hits']
            if actual_computations > 0:
                avg_computation_time = (self.stats['total_time'] - 
                                      self.stats['cache_hits'] * 0.001) / actual_computations
            else:
                avg_computation_time = 0.0
        else:
            cache_hit_rate = 0.0
            avg_time = 0.0
            avg_computation_time = 0.0
            
        return {
            'total_calls': self.stats['total_calls'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'subprocess_calls': self.stats['subprocess_calls'],
            'total_time': self.stats['total_time'],
            'avg_time_per_call': avg_time,
            'avg_computation_time': avg_computation_time,
            'cache_efficiency': cache_hit_rate * 100
        }

    def shutdown(self):
        """Clean up with performance reporting."""
        # Performance statistics are available via get_performance_stats() if needed
        # Removed automatic printing to reduce output clutter
        
        if self.workdir and os.path.exists(self.workdir):
            shutil.rmtree(self.workdir, ignore_errors=True)


class ParallelXTBProcessor:
    """
    Step 3 optimization: Parallel processing of multiple xTB calculations.
    Uses ThreadPoolExecutor for efficient parallel execution.
    """
    
    def __init__(self, cfg: XTBConfig, max_workers: int = 4, max_cache_size: int = 10000):
        self.cfg = cfg
        self.max_workers = max_workers
        # LRU cache with size limit
        self.cache: OrderedDict[str, float] = OrderedDict()
        self.max_cache_size = max_cache_size
        
        # Performance monitoring
        self.stats = {
            'total_batches': 0,
            'total_molecules': 0,
            'cache_hits': 0,
            'parallel_time': 0.0,
            'sequential_time': 0.0
        }

    def compute_energies_parallel(self, molecules_data: List[Tuple[List[str], np.ndarray]]) -> List[float]:
        """Compute multiple energies in parallel using ThreadPoolExecutor."""
        if not molecules_data:
            return []
        
        start_time = time.time()
        self.stats['total_batches'] += 1
        self.stats['total_molecules'] += len(molecules_data)
        
        # Check cache first and prepare uncached work
        results = [None] * len(molecules_data)
        uncached_indices = []
        
        for i, (atoms, coords) in enumerate(molecules_data):
            key = self._make_cache_key(atoms, coords)
            if key in self.cache:
                # Move to end (most recently used)
                energy = self.cache.pop(key)
                self.cache[key] = energy
                results[i] = energy
                self.stats['cache_hits'] += 1
            else:
                uncached_indices.append(i)
        
        # Process uncached items in parallel
        if uncached_indices:
            if len(uncached_indices) == 1:
                # Single item - use direct computation
                i = uncached_indices[0]
                atoms, coords = molecules_data[i]
                server = OptimizedXTBServer(self.cfg)
                try:
                    energy = server.compute_energy(atoms, coords)
                    results[i] = energy
                    key = self._make_cache_key(atoms, coords)
                    # Update cache with LRU eviction
                    if len(self.cache) >= self.max_cache_size:
                        self.cache.popitem(last=False)
                    self.cache[key] = energy
                finally:
                    server.shutdown()
            else:
                # Multiple items - use parallel processing
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(uncached_indices))) as executor:
                    # Submit tasks
                    future_to_index = {}
                    for i in uncached_indices:
                        atoms, coords = molecules_data[i]
                        future = executor.submit(self._compute_single_energy, atoms, coords)
                        future_to_index[future] = i
                    
                    # Collect results
                    for future in as_completed(future_to_index, timeout=60):
                        i = future_to_index[future]
                        try:
                            energy = future.result()
                            results[i] = energy
                            
                            # Update cache with LRU eviction
                            atoms, coords = molecules_data[i]
                            key = self._make_cache_key(atoms, coords)
                            if len(self.cache) >= self.max_cache_size:
                                self.cache.popitem(last=False)
                            self.cache[key] = energy
                            
                        except Exception as e:
                            print(f"Parallel computation failed for molecule {i}: {e}")
                            results[i] = float('inf')  # Error marker
        
        elapsed = time.time() - start_time
        self.stats['parallel_time'] += elapsed
        
        return results

    def _compute_single_energy(self, atoms: List[str], coords: np.ndarray) -> float:
        """Compute single energy in a separate thread."""
        server = OptimizedXTBServer(self.cfg)
        try:
            return server.compute_energy(atoms, coords)
        finally:
            server.shutdown()

    def _make_cache_key(self, atoms: List[str], coords: np.ndarray) -> str:
        """Create cache key consistent with OptimizedXTBServer."""
        m = hashlib.sha1()
        atoms_str = ",".join(atoms)
        m.update(atoms_str.encode('ascii'))
        rounded_coords = np.round(coords, decimals=2)
        m.update(rounded_coords.tobytes())
        return m.hexdigest()

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.stats['total_molecules'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_molecules']
            avg_batch_time = self.stats['parallel_time'] / max(self.stats['total_batches'], 1)
            throughput = self.stats['total_molecules'] / max(self.stats['parallel_time'], 0.001)
        else:
            cache_hit_rate = 0.0
            avg_batch_time = 0.0
            throughput = 0.0
            
        return {
            'total_batches': self.stats['total_batches'],
            'total_molecules': self.stats['total_molecules'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'avg_batch_time': avg_batch_time,
            'throughput': throughput,
            'parallel_time': self.stats['parallel_time']
        }

    def shutdown(self):
        """Clean up with performance reporting."""
        # Performance statistics are available via get_performance_stats() if needed
        # Removed automatic printing to reduce output clutter
        pass


# Enhanced worker pool using optimized servers
class OptimizedXTBWorkerPool:
    """Optimized worker pool using the enhanced XTB servers."""
    
    def __init__(self, xtb_cfg: XTBConfig, n_workers: int):
        import multiprocessing as mp

        self.n_workers = n_workers
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        self.next_task_id = 0
        
        # Use optimized worker process
        cfg_dict = {
            'charge': xtb_cfg.charge,
            'mult': xtb_cfg.mult,
            'method': xtb_cfg.method,
            'solvent': xtb_cfg.solvent,
            'executable': xtb_cfg.executable,
            'prefer_tblite': xtb_cfg.prefer_tblite,
        }
        
        for i in range(n_workers):
            p = mp.Process(target=optimized_xtb_worker_process,
                           args=(i, self.task_queue, self.result_queue, cfg_dict))
            p.start()
            self.workers.append(p)

    def submit(self, atoms: List[str], coords: np.ndarray) -> int:
        """Submit task for processing."""
        task_id = self.next_task_id
        self.next_task_id += 1
        self.task_queue.put((task_id, atoms, coords.copy()))
        return task_id

    def get_result(self) -> Tuple[int, Optional[float], Optional[str]]:
        """Get result from worker."""
        return self.result_queue.get()

    def shutdown(self):
        """Shutdown worker pool."""
        for _ in range(self.n_workers):
            self.task_queue.put(None)
        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


def optimized_xtb_worker_process(worker_id: int, task_queue, result_queue, xtb_cfg_dict: Dict):
    """Optimized worker process using enhanced XTB server."""
    cfg = XTBConfig(**xtb_cfg_dict)
    server = OptimizedXTBServer(cfg)
    
    try:
        while True:
            try:
                task = task_queue.get(timeout=0.1)  # Fast timeout
                if task is None:
                    break
                    
                task_id, atoms, coords = task
                
                try:
                    energy = server.compute_energy(atoms, coords)
                    result_queue.put((task_id, energy, None))
                except Exception as e:
                    result_queue.put((task_id, None, str(e)))
                    
            except Empty:
                continue
            except KeyboardInterrupt:
                break
    finally:
        server.shutdown()


# Backward compatibility
XTBServer = OptimizedXTBServer
XTBWorkerPool = OptimizedXTBWorkerPool
