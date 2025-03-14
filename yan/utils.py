import os
import sys
import torch
import torch.distributed as dist

class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()




def bench_kineto(fn, kernel_names, num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: str = None, barrier_comm_profiling: bool = False, flush_l2: bool = False):
    using_nsys = os.environ.get('YAN_NSYS_PROFILING', False)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output and not using_nsys else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) if not using_nsys else None
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) if not using_nsys else empty_suppress()
        with profiler:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((4096, 4096), dtype=torch.float, device='cuda')
                    rhs = torch.randn((4096, 4096), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(int(128e6 // 4), dtype=torch.int, device='cuda').zero_()
                    fn()
                
                if not using_nsys:
                    profiler.step()
    
    if using_nsys:
        return 1
    
    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tupled = isinstance(kernel_names, tuple)
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=300).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table: {prof_lines}'
        

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_times.append(float(time_str.replace(unit, '')) / scale)
                        break
                break
    return tuple(kernel_times) if is_tupled else kernel_times[0]

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def count_bytes(tensors):
    total = 0
    for t in tensors:
        if isinstance(t, tuple):
            total += count_bytes(t)
        else:
            total += t.numel() * t.element_size()
    return total