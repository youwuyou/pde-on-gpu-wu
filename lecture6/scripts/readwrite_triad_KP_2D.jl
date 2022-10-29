using CUDA
using BenchmarkTools

# define memcopy
# NOTE! using A = B just moves the pointer
@inbounds function memcopy_triad_KP!(A, B, C, s)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    A[ix,iy] = B[ix,iy] + s*C[ix,iy]
    return nothing
end


# see available GPUs
collect(devices())

# assign no more than one user per GPU
device!(1)   # select a GPU between 0-7

# using best nx=ny found on racklette
nx = ny = 16384

A = CUDA.zeros(Float64, nx, ny);
B = CUDA.rand(Float64, nx, ny);
C = CUDA.rand(Float64, nx, ny);
s = rand()

# finding out best thread number
max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
println("Max thread number = $(max_threads)")

thread_count = []
throughputs = []
threads_x = 32

for pow = 0: Int(log2(max_threads/threads_x))
		threads = (threads_x,2^pow)
		blocks = (nx÷threads[1], ny÷threads[2])
		t_it = @belapsed begin @cuda blocks=$blocks threads=$threads memcopy_triad_KP!($A, $B, $C, $s); synchronize() end
    T_tot = 3*1/1e9*nx*ny*sizeof(Float64)/t_it
    push!(thread_count, prod(threads))
    push!(throughputs, T_tot)
    println("(threads=$threads, blocks=$blocks) T_tot = $(T_tot)")
end

