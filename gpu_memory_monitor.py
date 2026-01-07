#!/usr/bin/env python3
"""
GPU Memory Monitor for Telly Spelly
Run this alongside telly-spelly to track VRAM usage over time.
"""

import subprocess
import time
import sys
from datetime import datetime

def get_gpu_memory():
    """Get current GPU memory usage using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, total, free = map(int, result.stdout.strip().split(', '))
            return {'used': used, 'total': total, 'free': free}
    except Exception as e:
        print(f"Error querying GPU: {e}")
    return None

def get_telly_gpu_memory():
    """Get GPU memory specifically used by telly-spelly processes"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            telly_mem = 0
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 2:
                    pid, mem = int(parts[0]), int(parts[1])
                    # Check if this PID is telly-spelly
                    try:
                        with open(f'/proc/{pid}/cmdline', 'r') as f:
                            cmdline = f.read()
                        if 'telly' in cmdline.lower() or 'whisper' in cmdline.lower():
                            telly_mem += mem
                    except (OSError, PermissionError, ProcessLookupError):
                        # Process may have exited or we don't have permission
                        pass
            return telly_mem
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return None

def main():
    print("GPU Memory Monitor for Telly Spelly")
    print("=" * 60)
    print("Monitoring GPU memory usage. Press Ctrl+C to stop.")
    print()

    baseline = None
    peak = 0
    samples = []

    try:
        while True:
            mem = get_gpu_memory()
            telly_mem = get_telly_gpu_memory()

            if mem:
                if baseline is None:
                    baseline = mem['used']

                peak = max(peak, mem['used'])
                delta = mem['used'] - baseline

                timestamp = datetime.now().strftime("%H:%M:%S")

                telly_str = f"  Telly: {telly_mem}MB" if telly_mem else ""

                # Color coding based on memory pressure
                if mem['used'] > mem['total'] * 0.9:
                    status = "CRITICAL"
                elif mem['used'] > mem['total'] * 0.75:
                    status = "HIGH"
                else:
                    status = "OK"

                print(f"[{timestamp}] Used: {mem['used']:5}MB / {mem['total']}MB  "
                      f"(+{delta:+5}MB from start)  Peak: {peak}MB  [{status}]{telly_str}")

                samples.append({
                    'time': time.time(),
                    'used': mem['used'],
                    'delta': delta
                })

                # Warn if memory is increasing steadily
                if len(samples) >= 10:
                    recent = samples[-10:]
                    if all(recent[i]['used'] <= recent[i+1]['used'] for i in range(len(recent)-1)):
                        if recent[-1]['used'] - recent[0]['used'] > 100:
                            print("  ^ WARNING: Memory usage steadily increasing!")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 60)
        print("Summary:")
        print(f"  Baseline: {baseline}MB")
        print(f"  Peak:     {peak}MB")
        if baseline:
            print(f"  Growth:   {peak - baseline}MB")
        if samples:
            print(f"  Samples:  {len(samples)}")

if __name__ == '__main__':
    main()
