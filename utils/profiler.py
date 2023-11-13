import functools

from torch.profiler import ProfilerActivity, profile, record_function


def profiler(name, sort_by="cpu_time_total"):
    def inner_profiler(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                with record_function(name):

                    output = func(*args, **kwargs)

            print(prof.key_averages().table(sort_by=sort_by, row_limit=10))

            return output

        return wrapper

    return inner_profiler
