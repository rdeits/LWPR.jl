module Lwpr

using Parameters

@with_kw struct RFParams{T}
    λ::T = 0.999
    D_def{Matrix{T}}
    γ::T = 1e-4
    α::T = 40
    w_gen::T = 0.1
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
    E::T
    H::T
    R::T

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
        p.E = zero(T)
        p.H = zero(T)
        p.R = zero(T)
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
    projections::Vector{Projection{T}}
    M::UpperTriangular{T, Matrix{T}}
    ∂J∂M::UpperTriangular{T, Matrix{T}}

    function ReceptiveField{T}(c::AbstractVector, Nout, params::RFParams) where T
        rf = new{T}(params)
        N = length(c)
        rf.x0 = zeros(T, N)
        rf.β0 = zero(T)
        rf.D = copy(params.D_def)
        rf.c = copy(c)
        rf.W = zero(T)
        rf.projections = [Projection{T}(N) for i in 1:2]
        C = cholfact(rf.D)
        rf.M = C[:U]
        rf.∂J∂M = UpperTriangular(zeros(N, N))
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
    @unpack λ, α, γ = rf.params
    ΣΣ∂J1∂w = zero(T)
    for k in eachindex(rf.projections)
        proj = rf.projections[k]
        # note that W has already been updated to W^{n+1} in _update_means!()
        proj.E = λ * proj.E + w * residuals[k]^2 # residuals[k] corresponds to res_{k-1} in the paper
        ΣΣ∂J1∂w += -E/rf.W^2 + 1/rf.W * (residuals[k]^2 - 2 * residuals[k + 1] * proj.s / proj.SS * proj.H - 2 * (proj.s / proj.SS)^2 * proj.R)
        h = w * proj.s^2 / proj.SS
        proj.H = λ * proj.H + w * proj.s * residuals[k] / (1 - h)
        proj.R = λ * proj.R + w^2 * residuals[k]^2 * proj.s^2 / (1 - h)
    end
    wJ2 = w / (rf.n * rf.W)

    n = size(rf.D, 1)
    Δx = x - rf.c
    for l in 1:n
        for r in 1:j
            ∂D∂Mrl = zeros(T, n, n)
            for j in 1:n
                for i in 1:n
                    if j == l
                        ∂D∂Mrl[i, j] += rf.M[r, i]
                    end
                    if i == l
                        ∂D∂Mrl[i, j] += rf.M[r, j]
                    end
                end
            end
            ∂w∂Mrl = -0.5 * w * Δx' * ∂D∂Mrl * Δx
            ∂J1∂Mrl = ∂w∂Mrl * ΣΣ∂J1∂w

            ∂J2∂Mrl = zero(T)
            for j in 1:n
                for i in 1:n
                    if j == l
                        ∂J2∂Mrl += rf.D[i, j] * rf.M[r, i]
                    end
                    if i == l
                        ∂J2∂Mrl += rf.D[i, j] * rf.M[r, j]
                    end
                end
            end
            ∂J2∂Mrl *= 2 * γ

            rf.∂J∂M[r, l] = ∂J1∂Mrl + ∂J2∂Mrl
        end
    end

    @.(rf.M -= α * rf.∂J∂M)
    rf.D = rf.M' * rf.M
end

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

struct Model{T}

    fields::Vector{ReceptiveField{T}}



end

