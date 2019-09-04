module Percolation

global const libpercolation = joinpath(@__DIR__, "..", "libpercolation.so")

using DelimitedFiles: readdlm, writedlm
using LinearAlgebra
using Statistics
using Base.Threads


"""
    tcm_square_lattice

Equivalent to `tcm_square_lattice_t` in "percolation.h":

```c
typedef struct tcm_square_lattice {
    int64_t (*neighbours)[4];
    int64_t length;
    int64_t size;
    bool    periodic;
} tcm_square_lattice_t;
```

It is a very low-level data type and should not be used directly. Consider using
[`SquareLattice`](@ref) instead.
"""
struct tcm_square_lattice
    neighbours::Ptr{Int64}
    length::Int64
    size::Int64
    periodic::Bool
end

"""
    tcm_cubic_lattice

Equivalent to `tcm_cubic_lattice_t` in "percolation.h":

```c
typedef struct tcm_cubic_lattice {
    int64_t (*neighbours)[6];
    int64_t length;
    int64_t size;
    bool    periodic;
} tcm_cubic_lattice_t;
```

It is a very low-level data type and should not be used directly. Consider using
[`CubicLattice`](@ref) instead.
"""
struct tcm_cubic_lattice
    neighbours::Ptr{Int64}
    length::Int64
    size::Int64
    periodic::Bool
end

"""
    tcm_triangular_lattice

Equivalent to `tcm_triangular_lattice_t` in "percolation.h":

```c
typedef struct tcm_triangular_lattice {
    int64_t (*neighbours)[6];
    int64_t length;
    int64_t length_y;
    int64_t size;
} tcm_triangular_lattice_t;
```

It is a very low-level data type and should not be used directly. Consider using
[`TriangularLattice`](@ref) instead.
"""
struct tcm_triangular_lattice
    neighbours::Ptr{Int64}
    length::Int64
    length_y::Int64
    size::Int64
end

"""
    tcm_triangular_stacked_lattice

Equivalent to `tcm_triangular_lattice_t` in "percolation.h":

```c
typedef struct tcm_triangular_stacked_lattice {
    int64_t (*neighbours)[8];
    int64_t length;
    int64_t length_y;
    int64_t size;
} tcm_triangular_stacked_lattice_t;
```

It is a very low-level data type and should not be used directly. Consider using
[`TriangularStackedLattice`](@ref) instead.
"""
struct tcm_triangular_stacked_lattice
    neighbours::Ptr{Int64}
    length::Int64
    length_y::Int64
    size::Int64
end

mutable struct tcm_random_number_generator
    raw::Ptr{Cvoid}

    function tcm_random_number_generator(n)
        p = ccall((:tcm_random_number_generator_init, libpercolation),
                  Ptr{tcm_random_number_generator}, (Cuint,), n)
        if p == C_NULL throw(OutOfMemoryError()) end
        generator = new(p)
        return finalizer(
            x -> ccall((:tcm_random_number_generator_deinit, libpercolation),
                       Cvoid, (Ptr{tcm_random_number_generator},), x.raw),
            generator)
    end
end

struct tcm_perc_results
    size::UInt32
    number_sites::Ptr{UInt32}
    number_clusters::Ptr{UInt32}
    max_cluster_size::Ptr{UInt32}
    has_wrapped_one::Ptr{UInt8}
    has_wrapped_two::Ptr{UInt8}
    chirality::Ptr{Float64}
    helicity::Ptr{Float64}
    magnetisation::Ptr{Float64}
end

struct tcm_percolation_results
    number_clusters::Ptr{Float64}
    max_cluster_size::Ptr{Float64}
    has_wrapped_one::Ptr{Float64}
    has_wrapped_two::Ptr{Float64}
    chirality::Ptr{Float64}
    helicity::Ptr{Float64}
    magnetisation::Ptr{Float64}
end

struct tcm_percolation_stats
    max_number_sites::Ptr{Float64}
    mean_number_sites::Ptr{Float64}
    max_number_children::Ptr{Float64}
    mean_number_children::Ptr{Float64}
end

struct tcm_convolution_state
    number_sites::Int64
    number_functions::Int64
    functions::Ptr{Ptr{Float64}}
    outputs::Ptr{Ptr{Float64}}
end


abstract type Lattice{N} end

"""
    SquareLattice <: Lattice{2}

A simple square lattice in two dimensions
----------------------------------------------------------------------

    SquareLattice(n, periodic=true)

Construct a new square lattice of side length `n` with (if `periodic == true`)
or without (if `periodic == false`) periodic boundary conditions.
"""
struct SquareLattice <: Lattice{2}
    neighbours::Array{Int64, 2}
    length::Int
    is_periodic::Bool

    function SquareLattice(n::Int, periodic::Bool=true)
        length_min = periodic ? 3 : 2
        if n < length_min
            throw(DomainError(n, "side length must be at least $length_min"))
        end
        lattice = new(Array{Int64, 2}(undef, 4, n^2), n, periodic)
        status = ccall((:tcm_compute_neighbours_square, libpercolation),
              Cint, (tcm_square_lattice,),
              tcm_square_lattice(pointer(lattice.neighbours), n, n^2, periodic))
        if status != 0 throw(SystemError("tcm_compute_neighbours_square", status)) end
        return lattice
    end
end

struct CubicLattice <: Lattice{3}
    neighbours::Array{Int64, 2}
    length::Int
    is_periodic::Bool

    function CubicLattice(n::Int, periodic::Bool=true)
        length_min = periodic ? 3 : 2
        if n < length_min
            throw(DomainError(n, "side length must be at least $length_min"))
        end
        lattice = new(Array{Int64, 2}(undef, 6, n^3), n, periodic)
        status = ccall((:tcm_compute_neighbours_cubic, libpercolation),
              Cint, (tcm_cubic_lattice,),
              tcm_cubic_lattice(pointer(lattice.neighbours), n, n^3, periodic))
        if status != 0 throw(SystemError("tcm_compute_neighbours_cubic", status)) end
        return lattice
    end
end

struct TriangularLattice <: Lattice{2}
    neighbours::Array{Int64, 2}
    length_x::Int
    length_y::Int

    function TriangularLattice(n::Int)
        if n < 2 throw(DomainError(n, "side length must be at least 2")) end
        length_x = n
        length_y = Int(round(n * 2 / sqrt(3.0)))
        lattice = new(Array{Int64, 2}(undef, 6, length_x * length_y), length_x, length_y)
        status = ccall((:tcm_compute_neighbours_triangular, libpercolation),
              Cint, (tcm_triangular_lattice,),
              tcm_triangular_lattice(pointer(lattice.neighbours),
                                     length_x, length_y, length_x * length_y))
        if status != 0 throw(SystemError("tcm_compute_neighbours_triangular", status)) end
        return lattice
    end
end

struct TriangularStackedLattice <: Lattice{3}
    neighbours::Array{Int64, 2}
    length::Int
    length_y::Int

    function TriangularStackedLattice(n::Int)
        if n < 2 throw(DomainError(n, "side length must be at least 2")) end
        length = n
        length_y = Int(round(n * 2 / sqrt(3.0)))
        lattice = new(Array{Int64, 2}(undef, 8, length * length * length_y),
                      length, length_y)
        status = ccall((:tcm_compute_neighbours_triangular_stacked, libpercolation),
            Cint, (tcm_triangular_stacked_lattice,),
            tcm_triangular_stacked_lattice(
                pointer(lattice.neighbours), length, length_y, length * length * length_y))
        if status != 0 throw(SystemError("tcm_compute_neighbours_triangular", status)) end
        return lattice
    end
end

number_sites(x::SquareLattice)::Int = x.length^2
number_sites(x::CubicLattice)::Int = x.length^3
number_sites(x::TriangularLattice)::Int = x.length_x * x.length_y
number_sites(x::TriangularStackedLattice)::Int = x.length * x.length * x.length_y

function tcm_lattice(x::SquareLattice)::tcm_square_lattice
    return tcm_square_lattice(
        pointer(x.neighbours), x.length, number_sites(x), x.is_periodic)
end

function tcm_lattice(x::CubicLattice)::tcm_cubic_lattice
    return tcm_cubic_lattice(
        pointer(x.neighbours), x.length, number_sites(x), x.is_periodic)
end

function tcm_lattice(x::TriangularLattice)::tcm_triangular_lattice
    return tcm_triangular_lattice(
        pointer(x.neighbours), x.length_x, x.length_y, number_sites(x))
end

function tcm_lattice(x::TriangularStackedLattice)::tcm_triangular_stacked_lattice
    return tcm_triangular_stacked_lattice(
        pointer(x.neighbours), x.length, x.length_y, number_sites(x))
end


mutable struct AverageAccumulator
    mu
    count::Int

    function AverageAccumulator(mu)
        fill!(mu, 0)
        return new(mu, 0)
    end
end

function call!(acc::AverageAccumulator, x)
    @assert length(acc.mu) == length(x)
    acc.count += 1
    BLAS.axpy!(-1.0, acc.mu, x)
    BLAS.axpy!(1.0 / acc.count, x, acc.mu)
end

function _percolate_impl(lattice::tcm_square_lattice,
                         out::tcm_perc_results,
                         rng::tcm_random_number_generator)::Nothing
    status = ccall((:tcm_percolate_square, libpercolation),
                   Cint, (Ref{tcm_square_lattice},
                          Ref{tcm_perc_results},
                          Ptr{tcm_percolation_stats},
                          Ptr{tcm_random_number_generator}),
                   lattice, out, C_NULL, rng.raw)
    if status != 0 throw(SystemError("tcm_percolate_square", status)) end
end

function percolate_test(xs, lattice::Lattice{N};
                        batch_size=1, out::Union{AbstractArray{Float64, 2}, Nothing}=nothing) where {N}
    if out == nothing out = Array{Float64, 2}(undef, length(xs), 4) end
    fill!(out, 0.0)

    number_clusters = Array{UInt32, 1}(undef, length(xs))
    max_cluster_size = Array{UInt32, 1}(undef, length(xs))
    has_wrapped_one = Array{UInt8, 1}(undef, length(xs))
    has_wrapped_two = Array{UInt8, 1}(undef, length(xs))

    results = tcm_perc_results(
        UInt32(length(xs)),
        pointer(xs),
        pointer(number_clusters),
        pointer(max_cluster_size),
        pointer(has_wrapped_one),
        pointer(has_wrapped_two),
        C_NULL,
        C_NULL,
        C_NULL)
    rng = tcm_random_number_generator(123)

    for i in 1:batch_size
        _percolate_impl(tcm_lattice(lattice), results, rng)
        view(out, :, 1) .+= number_clusters
        view(out, :, 2) .+= max_cluster_size
        view(out, :, 3) .+= has_wrapped_one
        view(out, :, 4) .+= has_wrapped_two
    end
    out ./= batch_size
    return out
end

function _percolate(n_min::Csize_t, n_max::Csize_t,
                    lattice::tcm_square_lattice,
                    out::tcm_percolation_results,
                    stats::Union{tcm_percolation_stats, Nothing})::Nothing
    status = ccall((:tcm_percolate_square, libpercolation),
        Cint, (Csize_t, Csize_t, tcm_square_lattice, Ref{tcm_percolation_results}, Ptr{tcm_percolation_stats}),
        n_min, n_max, lattice, out, stats != nothing ? Ref(stats) : C_NULL)
    if status != 0 throw(SystemError("tcm_percolate_square", status)) end
end

function _percolate(lattice::tcm_square_lattice, out::tcm_percolation_results)::Nothing
    status = ccall((:tcm_percolate_square, libpercolation),
        Cint, (tcm_square_lattice, Ref{tcm_percolation_results}, Ptr{tcm_percolation_stats}),
        lattice, out, C_NULL)
    if status != 0 throw(SystemError("tcm_percolate_square", status)) end
end

function _percolate(lattice::tcm_cubic_lattice, out::tcm_percolation_results)::Nothing
    status = ccall((:tcm_percolate_cubic, libpercolation),
        Cint, (tcm_cubic_lattice, Ref{tcm_percolation_results}), lattice, out)
    if status != 0 throw(SystemError("tcm_percolate_cubic", status)) end
end

function _percolate(lattice::tcm_triangular_lattice, out::tcm_percolation_results)::Nothing
    status = ccall((:tcm_percolate_triangular, libpercolation),
        Cint, (tcm_triangular_lattice, Ref{tcm_percolation_results}), lattice, out)
    if status != 0 throw(SystemError("tcm_percolate_triangular", status)) end
end

function _percolate(lattice::tcm_triangular_stacked_lattice, out::tcm_percolation_results)::Nothing
    status = ccall((:tcm_percolate_triangular_stacked, libpercolation),
        Cint, (tcm_triangular_stacked_lattice, Ref{tcm_percolation_results}), lattice, out)
    if status != 0 throw(SystemError("tcm_percolate_triangular", status)) end
end

function percolate(n_min, n_max, lattice::Lattice{N}, batch_size=1, profile=false;
                   out::Union{AbstractArray{Float64, 2}, Nothing}=nothing) where {N}
    if batch_size < 1 throw(DomainError(batch_size, "batch size must be at least 1")) end
    if N != 2
        throw(TypeError(:percolate,
            "only 2-dimensional lattices are supported.", Lattice))
    end
    n = number_sites(lattice)
    number_columns = 4
    if out == nothing out = Array{Float64, 2}(undef, n + 1, number_columns) end
    @assert size(out) == (n + 1, number_columns) # NOTE: This +1 is very important!!!
    acc  = AverageAccumulator(out)
    local_result = Array{Float64, 2}(undef, n + 1, number_columns)
    results = tcm_percolation_results(
        pointer(view(local_result, :, 1)),
        pointer(view(local_result, :, 2)),
        pointer(view(local_result, :, 3)),
        C_NULL,
        C_NULL,
        C_NULL,
        pointer(view(local_result, :, 4)),
    )

    if profile
        stats = Array{Float64, 2}(undef, n + 1, 4)
        stats_acc = AverageAccumulator(stats)
        local_stats = Array{Float64, 2}(undef, n + 1, 4)
        tcm_stats = tcm_percolation_stats(
            pointer(view(local_stats, :, 1)),
            pointer(view(local_stats, :, 2)),
            pointer(view(local_stats, :, 3)),
            pointer(view(local_stats, :, 4))
        )
    else
        tcm_stats = nothing
    end
    for i in 1:batch_size
        _percolate(Csize_t(n_min), Csize_t(n_max), tcm_lattice(lattice), results, tcm_stats)
        call!(acc, local_result)
        if profile call!(stats_acc, local_stats) end
    end
    fill!(view(out, 1:max(n_min-1, 1), :), 0.0)
    fill!(view(out, min(n_max+1, n + 1):(n + 1), :), 0.0)
    return profile ? (out, stats) : out
end

function percolate_parallel(lattice::Lattice{N}, batch_size=1) where {N}
    if batch_size < 1 throw(DomainError(batch_size, "batch size must be at least 1")) end
    if N != 2 && N != 3
        throw(TypeError(:percolate,
            "only 2- and 3-dimensional lattices are supported.", Lattice))
    end
    n = number_sites(lattice)
    number_workers = nthreads()
    number_columns = N == 2 ? 3 : 4
    @assert mod(batch_size, number_workers) == 0
    local_batch_size = div(batch_size, number_workers)
    results = Array{Float64, 3}(undef, n + 1, number_columns, number_workers)
    @threads for i in 1:number_workers
        percolate(lattice, local_batch_size; out=view(results, :, :, i))
    end
    mean = Array{Float64, 2}(undef, n + 1, number_columns)
    return Statistics.mean!(mean, results)
end

function convolution(measurements::AbstractArray{Float64, 2};
                     p_min::Float64=0.0, p_max::Float64=1.0)
    number_sites = size(measurements, 1) - 1
    N = size(measurements, 2)
    # Some sanity checks
    @assert number_sites > 0 "arrays must contain at least 2 elements"
    @assert 0.0 <= p_min && p_min <= p_max && p_max <= 1.0
    n_min = Int(floor(p_min * number_sites))
    n_max = Int(floor(p_max * number_sites))
    @assert n_min <= n_max
    result_size = n_max - n_min + Int32(1)
    # Constructing the convolution state
    results = Array{Float64, 2}(undef, result_size, N + 1)
    funcs::Array{Ptr{Float64}, 1} = [pointer(view(measurements, :, i)) for i in 1:N]
    outs::Array{Ptr{Float64}, 1} =
        [pointer(view(results, :, i)) - n_min * sizeof(Float64) for i in 1:N+1]
    state = tcm_convolution_state(
        Int64(number_sites), Int64(N), pointer(funcs), pointer(outs))
    status = ccall((:tcm_convolution, libpercolation),
        Cint, (Int64, Int64, Ref{tcm_convolution_state},),
        n_min, n_max, state)
    if status != 0 throw(SystemError("tcm_convolution", status)) end
    return results
end

# function convolution(measurements::NTuple{N, Array{Float64, 1}};
#                      p_min::Float64=0.0, p_max::Float64=1.0) where {N}
#     # Trivial case
#     if N == 0 return () end
#     number_sites = length(measurements[1]) - 1
#     # Some sanity checks
#     @assert number_sites > 0 "arrays must contain at least 2 elements"
#     for m in measurements
#         @assert length(m) == number_sites + 1 "all arrays must be of the same length"
#     end
#     @assert 0.0 <= p_min && p_min <= p_max && p_max <= 1.0
#     n_min = Int32(floor(p_min * number_sites))
#     n_max = Int32(floor(p_max * number_sites))
#     @assert n_min <= n_max
#     result_size = n_max - n_min + Int32(1)
#     # Constructing the convolution state
#     results::NTuple{N + 1, Array{Float64, 1}} = (
#         Array{Float64, 1}(undef, result_size),
#         [Array{Float64, 1}(undef, result_size) for m in measurements]...)
#     fns::Array{Ptr{Float64}, 1} = [pointer(m) for m in measurements]
#     outs::Array{Ptr{Float64}, 1} = [pointer(r) - n_min * sizeof(Float64) for r in results]
#     state = tcm_convolution_state(
#         Int32(number_sites), Int32(N), pointer(fns), pointer(outs))
#     status = ccall((:tcm_convolution, libpercolation),
#         Cint, (Int32, Int32, Ref{tcm_convolution_state},),
#         n_min, n_max, state)
#     if status != 0 throw(SystemError("tcm_convolution", status)) end
#     return results
# end

end # module
