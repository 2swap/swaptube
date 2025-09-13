## Shared functions for CUDA device (GPU) and Host (CPU)
These are all helper functions with precompiler directives that let them be used in both CUDA device code and regular CPU code.
Since they get compiled both by nvcc and the host compiler, I have just named them .c instead of .cu or .cpp.
