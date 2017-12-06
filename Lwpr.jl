module Lwpr

using Parameters

# @with_kw LwprParams


# struct Model

# @with_kw mutable struct ReceptiveField{T}
#     α::Matrix{T}
#     β::Vector{T}
#     β0::Vetor{T}
#     c::Vector{T}
#     D::Matrix{T}
#     H::Vector{T}
#     λ::Vector{T}
#     M::Matrix{T}
#     mean_x::Vector{T}
#     n_data::T
#     P::Matrix{T}
#     r::Vector{T}
#     s::Vector{T}
#     SSp::T
#     SSs2::Vector{T}
#     SSXres::Matrix{T}
#     SSYres::Matrix{T}
#     sum_e2::T
#     sum_e_cv2::Vector{T}
#     sum_w::Vector{T}
#     SxresYres::Matrix{T}
#     trustworthy::Bool
#     U::Matrix{T}
#     var_x::Vector{T}
# end

@with_kw struct RFParams{T}
    λ::T
    D_def{T}
end


mutable struct Projection{T}
    u::Vector{T}
    s::T
    SS::T
    SR::T
    SZ::Vector{T}
    SSE::T
    β::T
    p::Vector{T}
    H::T # TODO: is this a scalar? why is it bold in the paper?
    R::T # TODO: is this a scalar? why is it bold in the paper?

    function Projection{T}(N) where T
        p = new{T}()
        p.u = zeros(T, N)
        p.s = zero(T)
        p.SS = zero(T)
        p.SR = zero(T)
        p.SZ = zeros(T, N)
        p.SSE = zero(T)
        p.β = zero(T)
        p.p = zeros(T, N)
        p
    end
end


mutable struct ReceptiveField{T}
    params::RFParams{T}
    x0::Vector{T}  # estimate of the input mean
    β0::T
    D::Matrix{T}   # distance metric
    c::Vector{T}   # center of distribution
    W::T  # running total weight seen
    E::T
    projections::Vector{Projection{T}}

    function ReceptiveField{T}(c::AbstractVector, Nout, params::RFParams) where T
        rf = new{T}(params)
        N = length(c)
        rf.x0 = zeros(T, N)
        rf.β0 = zero(T)
        rf.D = copy(params.D_def)
        rf.c = copy(c)
        rf.W = zero(T)
        rf.E = zero(T)
        rf.projections = [Projection{T}(N) for i in 1:2]
        rf
    end
end

function activation(rf::ReceptiveField, x::AbstractVector)
    Δx = x .- rf.c
    exp(-0.5 * Δx' * rf.D * Δx)
end

function _update_means!(rf::ReceptiveField{T}, x::AbstractVector{T}, y::T, w) where T
    @unpack λ = rf.params
    Wnext = λ * rf.W + w
    rf.x0 .= ((λ * rf.W) .* rf.x0 .+ w .* x) ./ Wnext
    rf.β0 = (λ * rf.W * rf.β0 + w * y) / Wnext
    rf.W = Wnext
end

function _update_local_model!(rf::ReceptiveField{T}, residuals::Vector{T}, x::AbstractVector{T}, y::T, w) where T
    @unpack λ = rf.params
    z = x - rf.x0 # x0 already updated in _update_means!
    residuals[1] = y - rf.β0
    for i in eachindex(rf.projections)
        proj = rf.projections[i]
        @.(proj.u .= λ * proj.u + w * z * residuals[i]) # rf.residuals[i] corresponds to res_{i-1} in the paper
        proj.s = z' * proj.u
        proj.SS = λ * proj.SS + w * proj.s^2
        proj.SR = λ * proj.SR + w * proj.s^2 * residuals[i]
        @.(proj.SZ = λ * proj.SZ + w * z * proj.s)
        proj.β = proj.SR / proj.SS
        @.(proj.p = proj.SZ / proj.SS)
        @.(z = z - proj.s * proj.p)
        residuals[i + 1] = residuals[i] - proj.s * proj.β
        proj.res = res
        proj.SSE = λ * proj.SSE + w * res^2
    end
end

function _update_distance_metric!(rf::ReceptiveField{T}, residuals::Vector{T}, x::AbstractVector{T}, y::T, w) where T
    # note that W has already been updated to W^{n+1} in _update_means!()
    for k in eachindex(rf.projections)
        rf.E = λ * rf.E + w * residuals[k]^2 # residuals[k] corresponds to res_{k-1} in the paper



function predict(rf::ReceptiveField, x)
    y = rf.β0
    z = x - rf.x0
    for proj in rf.projections
        s = proj.u' * z
        y = y + proj.β * s
        @.(z = z - s * proj.p)
    end
    return y
end



function update!(rf::ReceptiveField{T}, x::AbstractVector{T}, y::T, w) where T
    _update_means!(rf, x, y, w)
    residuals = zeros(length(rf.projections) + 1) # TODO: pre-allocate
    _update_local_model!(rf, residuals, x, y, w)
    _update_distance_metric!(rf, residuals, x, y, w)
end



end

