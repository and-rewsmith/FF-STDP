from line_profiler import LineProfiler
import rl.benchmarks.src.change_detection2 as mod


def run():
    profiler = LineProfiler()
    profiler.add_function(mod.main)  # Add the functions you want to profile
    profiler.enable_by_count()       # Start profiling
    try:
        mod.main()                   # Call your module's main function
    finally:
        profiler.disable_by_count()
        profiler.print_stats()


if __name__ == "__main__":
    run()
