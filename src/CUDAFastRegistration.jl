module CUDAFastRegistration
using CUDA
using CUDAFastRegistrationLib_jll
import LinearAlgebra

export HDPcd,
    Register, create_hdpcd, create_cuda_fast_registrator, downsample_surfels,
    align_redux


include("types.jl")

include("NN_cuda.jl")


function clear_hypo(reg::Register)
    ccall(
        (:clear_hypotheses, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid},),
        reg.c_ptr,
    )
end

#
# void set_src_pcd_c(void* reg, HDPcd object_h);//void* register,
function set_src_pcd(reg::Register, object_h::HDPcd)
    ccall(
        (:set_src_pcd_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, HDPcd),
        reg.c_ptr,
        object_h,
    )
    reg.src_pcd = object_h
end

# void set_target_pcd_c(void* reg, HDPcd scene_h);
function set_target_pcd(reg::Register, scene_h::HDPcd)
    ccall(
        (:set_target_pcd_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, HDPcd),
        reg.c_ptr,
        scene_h,
    )
    reg.dst_pcd = scene_h
end

# void set_nn_feaure_k_c(void* reg, int correspondence_randomness);
function set_nn_feaure_k(reg::Register, correspondence_randomness::Int)
    ccall(
        (:set_nn_feaure_k_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Cint),
        reg.c_ptr,
        Int32(correspondence_randomness),
    )
end

# void set_max_correspondence_distance_c(void* reg, double max_correspondence_distance);
function set_max_correspondence_distance(
    reg::Register,
    max_correspondence_distance::Float64,
)
    ccall(
        (:set_max_correspondence_distance_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Cdouble),
        reg.c_ptr,
        max_correspondence_distance,
    )
end

# void set_min_inlier_fraction_c(void* reg, double inlier_fraction);
function set_min_inlier_fraction(reg::Register, inlier_fraction::Float64)
    ccall(
        (:set_min_inlier_fraction_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Cdouble),
        reg.c_ptr,
        inlier_fraction,
    )
end
# void set_min_inlier_number_c(void* reg, int inlier_number);
function set_min_inlier_number(reg::Register, inlier_number::Int)
    ccall(
        (:set_min_inlier_number_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Cint),
        reg.c_ptr,
        Int32(inlier_number),
    )
end
# void set_edge_similarity(void* reg, double edge_similarity);
function set_edge_similarity(reg::Register, edge_similarity::Float64)
    ccall(
        (:set_edge_similarity, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Cdouble),
        reg.c_ptr,
        edge_similarity,
    )
end
# bool cu_align_c(void* reg);
function cu_align_c(reg::Register)
    @assert false "we should not use this function for now. Use the following 3 instead."
    res = ccall((:cu_align_c, "libcuda_fast_registration"), Cint, (Ptr{Cvoid},), reg.c_ptr)
    return res
end

# void cuGenerateHypotheses_c(void* reg)
function cuGenerateHypotheses_c(reg::Register)::Nothing
    ccall(
        (:cuGenerateHypotheses_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid},),
        reg.c_ptr
    )
end

# void cuPolyRejection_c(void* reg, float edge_similarity)
function cuPolyRejection_c(reg::Register, edge_similarity::T)::Nothing where T<:Real
    ccall(
        (:cuPolyRejection_c, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Cfloat),
        reg.c_ptr,
        Cfloat(edge_similarity)
    )
end

# bool cudaRANSANC_c(void* reg)
function cudaRANSANC_c(reg::Register)::Bool
    int_res = ccall(
        (:cudaRANSANC_c, "libcuda_fast_registration"),
        Cint,
        (Ptr{Cvoid},),
        reg.c_ptr
    )
    return int_res != 0
end



function cu_align(
    reg::Register;
    nn_feature_k::Int64,
    max_correspondence_distance::Float64,
    inlier_fraction::Float64,
    inlier_number::Int64,
    edge_similarity::Float64,
) where {NT1<:Real, NT2<:Real}

    set_nn_feaure_k(reg, nn_feature_k)

    set_max_correspondence_distance(reg, max_correspondence_distance)
    # Inlier threshold
    set_min_inlier_fraction(reg, inlier_fraction)
    # Required inlier fraction for accepting a pose hypothesis
    set_min_inlier_number(reg, inlier_number)

    set_edge_similarity(reg, edge_similarity)
    
    cuGenerateHypotheses_c(reg)

    cuPolyRejection_c(reg, edge_similarity)
    converged::Bool = cudaRANSANC_c(reg)
    return converged
end


function cu_align(
    reg::Register,
    src::HDPcd,
    dst::HDPcd;
    nn_feature_k::Int64,
    max_correspondence_distance::Float64,
    inlier_fraction::Float64,
    inlier_number::Int64,
    edge_similarity::Float64,
    smart_swap::Bool = false
) where {NT1<:Real, NT2<:Real}
    swapped = false
    if smart_swap
        if src.count > dst.count
            swapped = true
            src, dst = dst, src
        end
    end
    reg.swapped = swapped
    set_src_pcd(reg, src)
    set_target_pcd(reg, dst)
    converged = cu_align(
        reg,
        nn_feature_k = nn_feature_k,
        max_correspondence_distance = max_correspondence_distance,
        inlier_fraction = inlier_fraction,
        inlier_number = inlier_number,
        edge_similarity = edge_similarity
    )
    T = Matrix{Float64}(LinearAlgebra.I, 4, 4)
    if converged
        T = get_final_transformation(reg)
        if swapped
            T = inv(T)
        end
    end
    return T, converged
end

# void get_final_transformation(void* reg, float* transfomation_ptr);
function get_final_transformation(reg::Register)
    transformation = Matrix{Float32}(undef, 4, 4)
    data_ptr = Base.unsafe_convert(Ptr{Cfloat}, transformation)
    ccall(
        (:get_final_transformation, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}),
        reg.c_ptr,
        data_ptr,
    )
    return convert(Matrix{Float64}, transformation)
end
# void get_target_infomation(void* reg, double* information_ptr);
function get_target_infomation(reg::Register)
    information = Matrix{Float64}(undef, 6, 6)
    data_ptr = Base.unsafe_convert(Ptr{Cdouble}, information)
    ccall(
        (:get_target_infomation, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        reg.c_ptr,
        data_ptr,
    )
    return information
end

# void get_src_information(void* reg, double* information_ptr);
function get_src_information(reg::Register)
    information = Matrix{Float64}(undef, 6, 6)
    data_ptr = Base.unsafe_convert(Ptr{Cdouble}, information)
    ccall(
        (:get_src_information, "libcuda_fast_registration"),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Cdouble}),
        reg.c_ptr,
        data_ptr,
    )
    return information
end

# void align_redux(void* reg, float* guess)
function align_redux(reg::Register, T_dst_src::Matrix{Float32})
    data_ptr = Base.unsafe_convert(Ptr{Cfloat}, T_dst_src)
    ccall(
        (:align_redux, :libcuda_fast_registration),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}),
        reg.c_ptr,
        data_ptr,
    )
    return
end
function align_redux(
    reg::Register,
    T_dst_src::Matrix{Float64};
    nn_feature_k::Int64,
    max_correspondence_distance::Float64,
    inlier_fraction::Float64,
    inlier_number::Int64,
)

    set_nn_feaure_k(reg, nn_feature_k)

    set_max_correspondence_distance(reg, max_correspondence_distance)
    # Inlier threshold
    set_min_inlier_fraction(reg, inlier_fraction)
    # Required inlier fraction for accepting a pose hypothesis
    set_min_inlier_number(reg, inlier_number)

    align_redux(reg, convert(Matrix{Float32}, T_dst_src))
end



# HDPcd downsample_convert_cpu_surfels_to_hdpcd_ptr(
#         float* h_surfel_data_ptr,
#         int64_t surfels_count,
#         int estimate_normal_bool,
#         double downsample_leaf_size,
#         double normal_radius,
#         double feature_radius)
function downsample_surfels(
    data::Vector{Float32},
    count::Union{UInt64,Int64};
    estimate_normal::Bool,
    downsample_leaf_size::Float64,
    normal_radius::Float64,
    feature_radius::Float64,
)
    @assert length(data) / 12 >= count
    hd_pcd = ccall(
        (:downsample_convert_cpu_surfels_to_hdpcd_ptr, :libcuda_fast_registration),
        HDPcd,
        (Ptr{Cfloat}, Clonglong, Cint, Cdouble, Cdouble, Cdouble),
        Base.unsafe_convert(Ptr{Cfloat}, data),
        Clonglong(count),
        Cint(estimate_normal),
        downsample_leaf_size,
        normal_radius,
        feature_radius,
    )
    finalizer(delete_hdpcd, hd_pcd)
    return hd_pcd
end

function downsample_surfels(
    data::CUDA.CuArray{Float32},
    count::Union{UInt64,Int64};
    estimate_normal::Bool,
    downsample_leaf_size::Float64,
    normal_radius::Float64,
    feature_radius::Float64,
)
    @assert length(data) / 12 >= count
    hd_pcd = ccall(
        (:downsample_convert_gpu_surfels_to_hdpcd_ptr, :libcuda_fast_registration),
        HDPcd,
        (CUDA.CuPtr{Cfloat}, Clonglong, Cint, Cdouble, Cdouble, Cdouble),
        data.ptr,
        Clonglong(count),
        Cint(estimate_normal),
        downsample_leaf_size,
        normal_radius,
        feature_radius,
    )
    finalizer(delete_hdpcd, hd_pcd)
    return hd_pcd
end

"""
downsample xyz and create HDPcd
"""
function downsample_xyz(
    data::AbstractArray{Float32},
    count::Union{UInt64,Int64},
    point_dim::Int64;
    estimate_normal::Bool=true,
    downsample_leaf_size::Float64,
    normal_radius::Float64,
    feature_radius::Float64,
    c_order::Bool=true
)
    @assert length(data) / point_dim == count
    @assert estimate_normal "we need to est normal here"
    @assert c_order "Currently only support c_order"
    # HDPcd downsample_convert_xyz_to_hdpcd_ptr(
    #     float* h_xyz_data_ptr,
    #     int64_t pt_count,
    #     int point_dim,
    #     int estimate_normal_bool,
    #     double downsample_leaf_size,
    #     double normal_radius,
    #     double feature_radius)
    hd_pcd = ccall(
        (:downsample_convert_xyz_to_hdpcd_ptr, :libcuda_fast_registration),
        HDPcd,
        (Ptr{Cfloat}, Clonglong, Cint, Cint, Cdouble, Cdouble, Cdouble),
        Base.unsafe_convert(Ptr{Cfloat}, data),
        Clonglong(count),
        Cint(point_dim),
        Cint(estimate_normal),
        downsample_leaf_size,
        normal_radius,
        feature_radius,
    )
    finalizer(delete_hdpcd, hd_pcd)
    return hd_pcd
end

"""
This is a util function
scale is the scale factor used to  (pcd_org - m) ./ scale
"""
function restore_scaled_T(
    T_dst_src_scaled::AbstractMatrix{T1},
    scale::T2,
    m_src::AbstractMatrix{T3},
    m_dst::AbstractMatrix{T3}
) where {T1<:Real, T2<:Real, T3<:Real}

    @assert size(m_src) == (3, 1)
    @assert size(m_dst) == (3, 1)

    T_dst_src_restored = Matrix{T1}(LinearAlgebra.I, 4, 4)

    Rˢ = T_dst_src_scaled[1:3, 1:3]
    tˢ = T_dst_src_scaled[1:3, 4:4]

    T_dst_src_restored[1:3, 1:3] = Rˢ
    T_dst_src_restored[1:3, 4:4] = m_dst - Rˢ * m_src + scale * tˢ

    return T_dst_src_restored
end

end # module
