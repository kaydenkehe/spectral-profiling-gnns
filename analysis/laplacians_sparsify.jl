using DelimitedFiles
using Laplacians
using LinearAlgebra
using Random
using SparseArrays

function read_upper_edges(path::String, n::Int)
    if filesize(path) == 0
        return spzeros(Float64, n, n)
    end

    edge_data = readdlm(path, ',', Float64)
    if ndims(edge_data) == 1
        edge_data = reshape(edge_data, 1, :)
    end

    src = Int.(edge_data[:, 1])
    dst = Int.(edge_data[:, 2])
    weights = Float64.(edge_data[:, 3])
    return sparse(vcat(src, dst), vcat(dst, src), vcat(weights, weights), n, n)
end

function write_upper_edges(path::String, adjacency)
    symmetric_adjacency = sparse((adjacency + adjacency') / 2)
    upper = triu(symmetric_adjacency, 1)
    rows, cols, weights = findnz(upper)

    open(path, "w") do io
        for i in eachindex(weights)
            if weights[i] != 0.0
                println(io, rows[i], ",", cols[i], ",", weights[i])
            end
        end
    end
end

if length(ARGS) != 5
    error("usage: julia laplacians_sparsify.jl INPUT_EDGES OUT_DIR NUM_NODES EPS_CSV SEED")
end

input_path = ARGS[1]
out_dir = ARGS[2]
num_nodes = parse(Int, ARGS[3])
epsilons = parse.(Float64, split(ARGS[4], ","))
seed = parse(Int, ARGS[5])

mkpath(out_dir)
adjacency = read_upper_edges(input_path, num_nodes)
stats_path = joinpath(out_dir, "sparsify_stats.csv")

open(stats_path, "w") do stats
    println(stats, "epsilon,sparsify_time_sec,sparse_edges,edge_file")

    for (idx, epsilon) in enumerate(epsilons)
        Random.seed!(seed + idx - 1)
        start_ns = time_ns()
        sparse_adjacency = Laplacians.sparsify(adjacency; ep=epsilon)
        elapsed_sec = (time_ns() - start_ns) / 1.0e9

        edge_file = joinpath(out_dir, "sparse_eps_" * replace(string(epsilon), "." => "p") * ".csv")
        write_upper_edges(edge_file, sparse_adjacency)
        sparse_edges = nnz(triu(sparse(sparse_adjacency), 1))

        println(stats, epsilon, ",", elapsed_sec, ",", sparse_edges, ",", edge_file)
    end
end
