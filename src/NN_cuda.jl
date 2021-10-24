# run nn search
"""
data
"""
function cuda_inrange(
    src_pcd::HDPcd,
    dst_pcd::HDPcd,
    dist_thres::Real;
    out::CUDA.CuArray{Int32}=CUDA.CuArray{Int32}(undef, src_pcd.count)
)
    ccall(
        (:nn_inrange_cuda_c, "libcuda_fast_registration"),
        Cvoid,
        (HDPcd, HDPcd, Cfloat, CUDA.CuPtr{Cint}),
        src_pcd,
        dst_pcd,
        Cfloat(dist_thres),
        out.ptr
    )
    return out

end


"""
1 indexed src to dst. The stored value is idex to dst
"""
function cuda_NN(
    src_pcd::HDPcd,
    dst_pcd::HDPcd;
    out::CUDA.CuArray{Int32}=CUDA.CuArray{Int32}(undef, src_pcd.count)
)
    ccall(
        (:feature_nn_cuda_c, "libcuda_fast_registration"),
        Cvoid,
        (HDPcd, HDPcd, CUDA.CuPtr{Cint}),
        src_pcd,
        dst_pcd,
        out.ptr
    )
    out .+= 1
    return out

end
