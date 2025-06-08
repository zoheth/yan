import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .tuner import jit_tuner
from .utils import compare_tensors

includes = ('"flash_attn/flash_attn_tk.cuh"',)
template = """
// Templated args from Python JIT call
constexpr auto d = {D};

flash_attn_func<d>(Q, K, V, O, batch_size, num_heads, seq_len, stream, timings);
"""


def flash_attn_tk(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: torch.Tensor, timings: torch.Tensor
) -> None:
    batch_size = Q.shape[0]
    num_heads = Q.shape[2]
    seq_len = Q.shape[1]
    d = Q.shape[3]

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    Output = Output.contiguous()

    # assert Q.shape == K.shape == V.shape == Output.shape
    # assert Q.dtype == K.dtype == V.dtype == Output.dtype == torch.bfloat16
    assert Q.device.type == "cuda"

    stream = torch.cuda.current_stream()

    global includes, template

    args = (Q, K, V, Output, batch_size, num_heads, seq_len, stream, timings)
    runtime = jit_tuner.compile_and_tune(
        name="flash_attn_func",
        keys={"D": d},
        space=(),
        includes=includes,
        arg_defs=(
            ("Q", torch.bfloat16),
            ("K", torch.bfloat16),
            ("V", torch.bfloat16),
            ("O", torch.bfloat16),
            ("batch_size", int),
            ("num_heads", int),
            ("seq_len", int),
            ("stream", torch.cuda.Stream),
            ("timings", torch.int32),
        ),
        template=template,
        args=args,
    )

    runtime(*args)

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

GPU_CLOCK_MHZ = 1695.0

def save_kernel_phase_gantt_chart(timings, name=None, verbose=False):
    if timings.device != torch.device('cpu'):
        timings = timings.cpu()

    kernel_phases = [
        {'start_idx': 0, 'end_idx': 1, 'label': 'Q Load & Setup', 'color': '#5DA271'},
        {'start_idx': 1, 'end_idx': 2, 'label': 'Launch Initial K/V Load', 'color': '#FEC601'},
        {'start_idx': 2, 'end_idx': 3, 'label': 'Attention Main Loop', 'color': '#FF312E'},
        {'start_idx': 3, 'end_idx': 4, 'label': 'Output Write', 'color': '#1F01B9'}
    ]

    timings_us = timings.float() / GPU_CLOCK_MHZ


    valid_starts = timings_us[:, 0][timings_us[:, 0] > 0]
    if len(valid_starts) == 0:
        print("No valid timing data found. Skipping chart generation.")
        return
        
    global_start_time = valid_starts.min()
    timings_us -= global_start_time

    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)
    num_processors = timings.shape[0]

    for proc in range(num_processors):
        if timings[proc, 0].item() <= 0:
            continue

        for phase in kernel_phases:
            start_idx, end_idx = phase['start_idx'], phase['end_idx']

            if timings[proc, start_idx].item() > 0 and timings[proc, end_idx].item() > 0:
                start_time = timings_us[proc, start_idx].item()
                end_time = timings_us[proc, end_idx].item()
                duration = end_time - start_time

                if duration > 0:
                    ax.barh(
                        proc,
                        duration,
                        left=start_time,
                        height=0.8,
                        color=phase['color'],
                        alpha=0.75,
                        edgecolor='black',
                        linewidth=0.5
                    )

    ax.set_xlabel('Time (microseconds)', fontsize=12)
    ax.set_ylabel('Processor ID (SMID)', fontsize=12)
    ax.set_title(f'Kernel Execution Timeline by Phase (Clock: {GPU_CLOCK_MHZ} MHz)', fontsize=16)
    ax.set_yticks(range(0, num_processors, 8))
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.invert_yaxis()

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=phase['color'], alpha=0.75,
                      label=phase['label'])
        for phase in kernel_phases
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 1])


    timestamp = int(time.time())
    filename = f'kernel_timeline_{name}_{timestamp}.png' if name else f'kernel_timeline_{timestamp}.png'
    plt.savefig(filename, dpi=200)
    print(f"Gantt chart saved as {filename}")

    if verbose:
        print("\n--- Kernel Phase Timing Statistics (microseconds) ---")
        total_duration_us = timings_us[:, 4] - timings_us[:, 0]
        valid_durations = total_duration_us[total_duration_us > 0]
        
        if len(valid_durations) > 0:
            print(f"Total execution time (max duration of any processor): {valid_durations.max():.3f} µs")
            print(f"Average processor execution time: {valid_durations.mean():.3f} µs\n")
            
            for phase in kernel_phases:
                durations = timings_us[:, phase['end_idx']] - timings_us[:, phase['start_idx']]
                valid_phase_durations = durations[durations > 0]
                if len(valid_phase_durations) > 0:
                    mean_time = valid_phase_durations.mean()
                    print(f"Phase '{phase['label']}': Average duration = {mean_time:.3f} µs")
        else:
            print("No valid durations to calculate statistics.")

def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        shape = (32, 1024, 1, 64)  # (batch_size, seq_len, num_heads, head_dim)
        dtype = torch.bfloat16
        
        Q = torch.randn(*shape, device="cuda", dtype=dtype)
        K = torch.randn(*shape, device="cuda", dtype=dtype)
        V = torch.randn(*shape, device="cuda", dtype=dtype)
        Output = torch.zeros(*shape, device="cuda", dtype=dtype)
        
        timings = torch.zeros([128, 64], device="cuda", dtype=torch.int32)
        
        flash_attn_tk(Q, K, V, Output, timings)

        Q = Q.permute(0, 2, 1, 3).contiguous()
        K = K.permute(0, 2, 1, 3).contiguous()
        V = V.permute(0, 2, 1, 3).contiguous()
        Output = Output.permute(0, 2, 1, 3)
        expected_output = F.scaled_dot_product_attention(
            Q, K, V
        )

        passed = compare_tensors( Output, expected_output, mode="similarity", diff_tolerance=1e-4)

        save_kernel_phase_gantt_chart(timings, name="attend_ker", verbose=True)
        # if(not passed):
        #     print("Expected output:")
        #     print(expected_output[31, 63, 1023, :10])
        #     print("Actual output:")
        #     print(Output[31, 63, 1023, :10])