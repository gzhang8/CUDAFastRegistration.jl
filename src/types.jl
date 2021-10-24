
"""
struct HDPcd{
  // bool is_data_owner_;
  int64_t count_;
  float* pcd_h_; # x1, x2, x3, x4...| y1, y2, y3, ... | z1 ...| normal_x, |ny|nz
  float* pcd_d_;

  // store fp16 data only in host so that it can be stored to a grid quickly and then memcp tp GPU
  int * pcd_fp16_h_;
  // for cuda feature NN search search
  float* feature_d_;
  size_t feature_pitch_in_float_;
};
"""
mutable struct HDPcd
    count::Int64
    pcd_h::Ptr{Float32}
    pcd_d::Ptr{Float32}
    pcd_fp16_h::Ptr{Int32}
    feature_d::Ptr{Float32}
    feature_pitch_in_float::Csize_t
end

# HDPcd create_hdpcd_from_ptrs (float* xyz,
#                     float* normal_xyz,
#                     float* feature_vectors,
#                     int64_t count);

function create_hdpcd(
    xyz::Matrix{Float64},
    normal::Matrix{Float64},
    feature::Matrix{Float32},
)
    count = size(xyz, 2)

    xyz32 = convert(Matrix{Float32}, xyz)
    normal32 = convert(Matrix{Float32}, normal)

    xyz_ptr = Base.unsafe_convert(Ptr{Float32}, xyz32)
    normal_ptr = Base.unsafe_convert(Ptr{Float32}, normal32)
    feature_ptr = Base.unsafe_convert(Ptr{Float32}, feature)
    hdpcd = ccall(
        (:create_hdpcd_from_ptrs, "libcuda_fast_registration"),
        HDPcd,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Clonglong),
        xyz32,
        normal32,
        feature_ptr,
        count,
    )

    finalizer(delete_hdpcd, hdpcd)
    return hdpcd


end

function create_hdpcd(
    xyz32::Matrix{Float32},
    normal32::Matrix{Float32},
    feature::Matrix{Float32},
)
    count = size(xyz32, 2)
    @assert size(normal32, 2) == size(feature, 2) == count


    xyz_ptr = Base.unsafe_convert(Ptr{Float32}, xyz32)
    normal_ptr = Base.unsafe_convert(Ptr{Float32}, normal32)
    feature_ptr = Base.unsafe_convert(Ptr{Float32}, feature)
    hdpcd = ccall(
        (:create_hdpcd_from_ptrs, "libcuda_fast_registration"),
        HDPcd,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Clonglong),
        xyz32,
        normal32,
        feature_ptr,
        count,
    )

    finalizer(delete_hdpcd, hdpcd)
    return hdpcd


end


# void delete_hdpcd (HDPcd pcd);
function delete_hdpcd(hdpcd::HDPcd)
    ccall((:delete_hdpcd, "libcuda_fast_registration"), Cvoid, (HDPcd,), hdpcd)
end

# float* getFeaturePtr_h(HDPcd pcd);
function get_feature_mat(hdpcd::HDPcd)::Matrix{Float32}
    fea_ptr = ccall(
        (:getFeaturePtr_h, "libcuda_fast_registration"),
        Ptr{Cfloat},
        (HDPcd,),
        hdpcd
    )
    FEATURE_DIM = 33
    fea_mat = Base.unsafe_wrap(
        Matrix{Float32},
        fea_ptr,
        (hdpcd.count, FEATURE_DIM),
        own=true)
    return transpose(fea_mat)
end

function get_point_mat(
    hdpcd::HDPcd;
    xyz_only::Bool=false,
    per_col::Bool=false,
    copy::Bool=false
)#::Matrix{Float32}
    pcd_mat = Base.unsafe_wrap(
        Matrix{Float32},
        hdpcd.pcd_h,
        (hdpcd.count, 6),
        own=false
    )
    pcd_mat = if xyz_only
        pcd_mat[:, 1:3]
    else
        pcd_mat
    end
    pcd_mat = if per_col
        transpose(pcd_mat)
    else
        pcd_mat
    end

    pcd_mat = if copy
        Base.copy(pcd_mat)
    else
        pcd_mat
    end
    return pcd_mat
end



mutable struct Register
    swapped::Bool
    c_ptr::Ptr{Cvoid}
    src_pcd::Union{HDPcd, Nothing}
    dst_pcd::Union{HDPcd, Nothing}
    src_feature::Union{Matrix{Float32}, Nothing}
    dst_feature::Union{Matrix{Float32}, Nothing}
    N_hypo::Int64
    sample_size::Int64
    rand_nums_cpu::Matrix{Float64}
    hypotheses_cu::CUDA.CuArray{Int32} # mapped array from c++ cuda
    rand_tmp_space_cu::CUDA.CuArray{Float64}
    levy_out_tmp_cu::CUDA.CuArray{Int32}
end

# void* create_cuda_fast_registrator_c(int64_t hypotheses_count,
#                    int sample_size,
#                    float cell_edge_len,
#                    float cube_edge_len,
#                    int idx_per_cell);
function create_cuda_fast_registrator(
    hypotheses_count::Int64,
    sample_size::Int,
    cell_edge_len::Float64,
    cube_edge_len::Float64,
    idx_per_cell::Int,
)

    ptr = ccall(
        (:create_cuda_fast_registrator_c, "libcuda_fast_registration"),
        Ptr{Cvoid},
        (Clonglong, Cint, Cfloat, Cfloat, Cint),
        hypotheses_count,
        Int32(sample_size),
        Float32(cell_edge_len),
        Float32(cube_edge_len),
        Int32(idx_per_cell),
    )

    # get hypothese cuda storage and map it to CuArray

    hopo_cu_ptr::CUDA.CuPtr{Cint}= ccall(
        (:get_hypotheses_d, "libcuda_fast_registration"),
        CUDA.CuPtr{Cint},
        (Ptr{Cvoid}, ),
        ptr
    )
    hypo_cu = Base.unsafe_wrap(
        CUDA.CuArray{Int32},
        hopo_cu_ptr,
        hypotheses_count * sample_size * 2, # src dst
        own=false
    )

    rand_tmp_space_cu::CUDA.CuArray{Float64} = CUDA.CuArray{Float64}(
        undef, hypotheses_count * sample_size
    )
    levy_out_tmp_cu =CUDA.CuArray{Int32}(undef, hypotheses_count*sample_size)

    swapped = false
    reg_jl = Register(swapped, ptr, nothing, nothing, nothing, nothing,
        hypotheses_count, Int64(sample_size), Matrix{Float64}(undef, 0, 0),
        hypo_cu, rand_tmp_space_cu, levy_out_tmp_cu
    )
    finalizer(delete_cuda_fast_registration, reg_jl)
    return reg_jl
end



# void delete_cuda_fast_registration_c(void* reg);
function delete_cuda_fast_registration(reg::Register)
    ccall(
        (:delete_cuda_fast_registration_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid},),
        reg.c_ptr,
    )
end
