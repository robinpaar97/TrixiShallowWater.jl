# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

####################################################################################################
# 2D Shallow Water Linearized Moment Equations
#####################################################################################################


@doc raw"""
    ShallowWaterLinearizedMomentEquations2D(; gravity, H0 = zero(gravity), n_moments,
                                    nu = 0.1, lambda = 0.1, rho = 1000.0,
                                    nman = 0.0165)
Shallow water linearized moment equations in two spatial dimensions. The equations are given by
```math
\begin{cases}
\partial_t h + \partial_x hv1 + \partial_y hv2 = 0, \\
\partial_t hv1 + \partial_x \left(hv1^2 + h\sum\limits_{i=1}^N \frac{\alpha_i^2}{2i+1} \right) + \partial_y \left(hv1v2 + h\sum\limits_{i=1}^N \frac{\alpha_i\beta_i}{2i+1} \right) = -gh\partial_x(h+b), \\
\partial_t hv2 + \partial_x \left(hv1v2 + h\sum\limits_{i=1}^N \frac{\alpha_i\beta_i}{2i+1} \right) + \partial_y \left(hv2^2 + h\sum\limits_{i=1}^N \frac{\beta_i^2}{2i+1} \right) = -gh\partial_y(h+b), \\
\partial_t h\alpha_i + \partial_x 2hv1\alpha_i + \partial_y(hv1\beta_i + hv2\alpha_i) =  v1(\partial_x h\alpha_i + \partial_y h\beta_i), \\
\partial_t h\beta_i + \partial_x(hv1\beta_i + hv2\alpha_2) + \partial_y 2hv2\beta_i =  v2(\partial_x h\alpha_i + \partial_y h\beta_i),
\end{cases}
```

The unknown quantities are the water and sediment height ``h``, the two velocity components ``v1`` and ``v2`` as well as the moments
``\alpha_i, \beta_i`` for ``i = 1, ..., n_{moments}``. The terms ``A_{ijk}`` and ``B_{ijk}`` are moment tensors that
are precomputed using shifted Legendre polynomials and ``g`` is the gravitational acceleration.

The conservative variable water height ``h`` is measured from the bottom topography ``b``, therefore
one also defines the total water height as ``H = h + b``.

The additional quantity ``H_0`` is also available to store a reference value for the total water
height that is useful to set initial conditions or test the "lake-at-rest" well-balancedness.

Additionally the system can be extended with two different friction models like in the 1D case.
A Newtonian slip friction model is available that uses slip length ``\lambda`` and kinematic viscosity ``\nu`` as parameters.
The source term for this friction model is given by [`source_term_newtonian_slip_friction`](@ref).
A Manning friction model is also available that uses the Manning roughness coefficient `nman` and
fluid density `rho` as a parameter.
The source term for this friction model is given by [`source_term_manning_friction`](@ref).

For details to the 1D case see the paper:
- Julio Careaga, Patrick Ersing, Julian Koellermeier, Andrew R. Winters (2026)
  Entropy analysis and entropy stable DG methods for the shallow water moment equations
  [DOI: 10.48550/arXiv.2602.06513](https://doi.org/10.48550/arXiv.2602.06513)
An extension to the 2D case with more details is given by the master thesis:
- Rima Said (RWTH Aachen University) (2026)
  Two-Dimensional Shallow Water Moment Equations: Entropy Analysis and Entropy Stable DG Methods
  [DOI: ]() 
"""

# Same conservative/primitive variable layout as ShallowWaterMomentEquations2D, only difference from the
# nonlinear model is that the A and B moment tensors are dropped and only C stays, since it's only used in the friction
# source terms, not the flux.

struct ShallowWaterLinearizedMomentEquations2D{NVARS, NMOMENTS, RealT <: Real,
                                               Array2 <: AbstractArray{RealT, 2}} <:
       AbstractMomentEquations{2, NVARS, NMOMENTS}
    gravity::RealT   # gravitational acceleration
    H0::RealT        # constant "lake-at-rest" total water height
    n_moments::Integer  # number of moments (per direction)
    # Moment matrix for the friction term (shared between the a- and c-moment blocks), n_moments-sized.
    C::Array2
    # Friction related quantities
    nu::RealT       # kinematic viscosity
    lambda::RealT  # slip length
    rho::RealT    # fluid density (relevant for Manning friction)
    nman::RealT   # Manning roughness coefficient

    function ShallowWaterLinearizedMomentEquations2D{NVARS, NMOMENTS, RealT, Array2}(gravity::RealT,
                                                                                      H0::RealT,
                                                                                      n_moments::Integer,
                                                                                      C::Array2,
                                                                                      nu::RealT,
                                                                                      lambda::RealT,
                                                                                      rho::RealT,
                                                                                      nman::RealT) where {
                                                                                                          NVARS,
                                                                                                          NMOMENTS,
                                                                                                          RealT <:
                                                                                                          Real,
                                                                                                          Array2 <:
                                                                                                          AbstractArray{RealT,
                                                                                                                        2}
                                                                                                          }
        new(gravity, H0, n_moments, C, nu, lambda, rho, nman)
    end
end

# See ShallowWaterMomentEquations2D for the same NMOMENTS/NVARS convention: NMOMENTS is the
# doubled type parameter (2*n_moments), NVARS = h, hv1, hv2, a_1..a_n, c_1..c_n, b.
function ShallowWaterLinearizedMomentEquations2D(; gravity, H0 = zero(gravity), n_moments,
                                                  nu = 0.1, lambda = 0.1, rho = 1000.0,
                                                  nman = 0.0165)
    RealT = promote_type(typeof(gravity), typeof(H0))

    NMOMENTS = n_moments
    NVARS = 2 * NMOMENTS + 4

    C = compute_C_matrix(n_moments)
    Array2 = typeof(C)

    return ShallowWaterLinearizedMomentEquations2D{NVARS, NMOMENTS, RealT, Array2}(gravity,
                                                                                    H0,
                                                                                    n_moments,
                                                                                    C,
                                                                                    nu,
                                                                                    lambda,
                                                                                    rho,
                                                                                    nman)
end

@inline function Base.real(::ShallowWaterLinearizedMomentEquations2D{NVARS, NMOMENTS,
                                                                      RealT, Array2}) where {
                                                                                             NVARS,
                                                                                             NMOMENTS,
                                                                                             RealT <:
                                                                                             Real,
                                                                                             Array2 <:
                                                                                             AbstractArray{RealT,
                                                                                                           2}
                                                                                             }
    RealT
end

Trixi.have_nonconservative_terms(::ShallowWaterLinearizedMomentEquations2D) = True()

function Trixi.varnames(::typeof(cons2cons),
                        equations::ShallowWaterLinearizedMomentEquations2D)
    conservative_moments_a = ntuple(n -> "ha" * string(n), Val(nmoments(equations)))
    conservative_moments_c = ntuple(n -> "hc" * string(n), Val(nmoments(equations)))

    return ("h", "hv1", "hv2", conservative_moments_a..., conservative_moments_c..., "b")
end

function Trixi.varnames(::typeof(cons2prim),
                        equations::ShallowWaterLinearizedMomentEquations2D)
    primitive_moments_a = ntuple(n -> "a" * string(n), Val(nmoments(equations)))
    primitive_moments_c = ntuple(n -> "c" * string(n), Val(nmoments(equations)))
    return ("H", "v1", "v2", primitive_moments_a..., primitive_moments_c..., "b")
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t, surface_flux_functions,
                                 equations::ShallowWaterLinearizedMomentEquations2D)

Identical construction to the nonlinear 2D slip wall: reflects the normal component of `(hv1, hv2)`
and of every moment pair `(a_i, c_i)`, keeps tangential components unchanged. Duplicated here
(rather than shared via Union) to match how `shallow_water_linearized_moments_1d.jl` redefines its
own `boundary_condition_slip_wall` rather than reusing the nonlinear one, the body is the same
in both files, only the dispatch type differs.
"""
@inline function Trixi.boundary_condition_slip_wall(u_inner,
                                                    orientation::Integer,
                                                    direction,
                                                    x,
                                                    t,
                                                    surface_flux_functions,
                                                    equations::ShallowWaterLinearizedMomentEquations2D)
    surface_flux_function, nonconservative_flux_function = surface_flux_functions

    h = u_inner[1]
    hv1 = u_inner[2]
    hv2 = u_inner[3]
    ha, hc = moments(u_inner, equations)
    b = u_inner[end]

    if orientation == 1
        u_boundary = SVector(h, -hv1, hv2, (-ha)..., hc..., b)
    else
        u_boundary = SVector(h, hv1, -hv2, ha..., (-hc)..., b)
    end

    if iseven(direction)
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
        noncons_flux = nonconservative_flux_function(u_inner, u_boundary, orientation,
                                                     equations)
    else
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
        noncons_flux = nonconservative_flux_function(u_inner, u_boundary, orientation,
                                                     equations)
    end
    return flux, noncons_flux
end

# Calculate 2D advective portion of the flux for a single point.
# Same structure as ShallowWaterMomentEquations2D's flux, minus the equations.A tensor
# contraction, mirrors how the 1D linearized flux drops the A[i,j,k] loop and keeps only
# `f_moments = 2 * ha * v`.
@inline function Trixi.flux(u,
                            orientation::Integer,
                            equations::ShallowWaterLinearizedMomentEquations2D)
    h = waterheight(u, equations)
    hv1 = u[2]
    hv2 = u[3]
    ha, hc = moments(u, equations)

    v1, v2 = velocities(u, equations)
    a = ha / h
    c = hc / h

    if orientation == 1
        f1 = hv1
        f2 = hv1 * v1
        f3 = hv1 * v2
        for i in eachmoment(equations)
            f2 += ha[i] * a[i] / (2 * i + 1)
            f3 += ha[i] * c[i] / (2 * i + 1)
        end
        f_moments1 = 2 * ha * v1
        f_moments2 = hc * v1 + ha * v2
    else
        f1 = hv2
        f2 = hv1 * v2
        f3 = hv2 * v2
        for i in eachmoment(equations)
            f2 += a[i] * hc[i] / (2 * i + 1)
            f3 += hc[i] * c[i] / (2 * i + 1)
        end
        f_moments1 = hc * v1 + ha * v2
        f_moments2 = 2 * hc * v2
    end

    return SVector{2 * nmoments(equations) + 4, real(equations)}(f1, f2, f3, f_moments1...,
                                                                  f_moments2..., 0)
end

"""
    flux_ec(u_ll, u_rr, orientation::Integer, equations::ShallowWaterLinearizedMomentEquations2D)

Same as `flux_ec` for `ShallowWaterMomentEquations2D`, minus the `equations.A[i, j, k]`
contraction loop,  the linearized model has no moment-moment tensor coupling.
"""
@inline function flux_ec(u_ll,
                         u_rr,
                         orientation::Integer,
                         equations::ShallowWaterLinearizedMomentEquations2D)
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    hv1_ll = u_ll[2]
    hv1_rr = u_rr[2]
    hv2_ll = u_ll[3]
    hv2_rr = u_rr[3]
    ha_ll, hc_ll = moments(u_ll, equations)
    ha_rr, hc_rr = moments(u_rr, equations)

    v1_ll, v2_ll = velocities(u_ll, equations)
    v1_rr, v2_rr = velocities(u_rr, equations)
    a_ll = ha_ll / h_ll
    a_rr = ha_rr / h_rr
    c_ll = hc_ll / h_ll
    c_rr = hc_rr / h_rr

    v1_avg = 0.5 * (v1_ll + v1_rr)
    hv1_avg = 0.5 * (hv1_ll + hv1_rr)
    v2_avg = 0.5 * (v2_ll + v2_rr)
    hv2_avg = 0.5 * (hv2_ll + hv2_rr)
    a_avg = 0.5 * (a_ll + a_rr)
    c_avg = 0.5 * (c_ll + c_rr)
    ha_avg = 0.5 * (ha_ll + ha_rr)
    hc_avg = 0.5 * (hc_ll + hc_rr)

    if orientation == 1
        f1 = hv1_avg
        f2 = hv1_avg * v1_avg
        f3 = hv1_avg * v2_avg
        for i in eachmoment(equations)
            f2 += ha_avg[i] * a_avg[i] / (2 * i + 1)
            f3 += ha_avg[i] * c_avg[i] / (2 * i + 1)
        end
        f_moments1 = hv1_avg * a_avg + ha_avg * v1_avg
        f_moments2 = hv1_avg * c_avg + ha_avg * v2_avg
    else
        f1 = hv2_avg
        f2 = hv1_avg * v2_avg
        f3 = hv2_avg * v2_avg
        for i in eachmoment(equations)
            f2 += a_avg[i] * hc_avg[i] / (2 * i + 1)
            f3 += hc_avg[i] * c_avg[i] / (2 * i + 1)
        end
        f_moments1 = v1_avg * hc_avg + a_avg * hv2_avg
        f_moments2 = hv2_avg * c_avg + hc_avg * v2_avg
    end

    return SVector{2 * nmoments(equations) + 4, real(equations)}(f1, f2, f3, f_moments1...,
                                                                  f_moments2..., 0)
end

"""
    flux_nonconservative_ec(u_ll, u_rr, orientation::Integer,
                             equations::ShallowWaterLinearizedMomentEquations2D)

Same as `flux_nonconservative_ec` for `ShallowWaterMomentEquations2D`, minus the
`equations.B[i, j, k]` contraction loop.
"""
@inline function flux_nonconservative_ec(u_ll,
                                         u_rr,
                                         orientation::Integer,
                                         equations::ShallowWaterLinearizedMomentEquations2D)
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    ha_ll, hc_ll = moments(u_ll, equations)
    ha_rr, hc_rr = moments(u_rr, equations)
    b_ll = u_ll[end]
    b_rr = u_rr[end]

    v1_ll, v2_ll = velocities(u_ll, equations)

    h_jump = h_rr - h_ll
    b_jump = b_rr - b_ll
    ha_jump = ha_rr - ha_ll
    hc_jump = hc_rr - hc_ll

    g = equations.gravity

    if orientation == 1
        f2 = g * h_ll * (h_jump + b_jump)
        f3 = 0
        f_moments1 = -v1_ll * ha_jump
        f_moments2 = -v2_ll * ha_jump
    else
        f2 = 0
        f3 = g * h_ll * (h_jump + b_jump)
        f_moments1 = -v1_ll * hc_jump
        f_moments2 = -v2_ll * hc_jump
    end

    return SVector{2 * nmoments(equations) + 4, real(equations)}(0, f2, f3, f_moments1...,
                                                                  f_moments2..., 0)
end

# Duplicated (not Union-shared) to mirror shallow_water_linearized_moments_1d.jl, which
# redefines this functor per-type even though the body is identical to the nonlinear one 
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll,
                                                              u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::ShallowWaterLinearizedMomentEquations2D)
    λ = dissipation.max_abs_speed(u_ll,
                                  u_rr,
                                  orientation_or_normal_direction,
                                  equations)
    diss = -0.5 * λ * (u_rr - u_ll)
    return SVector(@views diss[1:(end - 1)]..., zero(eltype(u_ll)))
end

# NOTE: no separate DissipationLaxFriedrichsEntropyVariables method here, exactly like
# shallow_water_linearized_moments_1d.jl, the one defined in shallow_water_moments_2d.jl already
# covers this type via its Union, since that H-matrix construction never references A or B.
@inline function Trixi.max_abs_speeds(u, equations::ShallowWaterLinearizedMomentEquations2D)
    h = waterheight(u, equations)
    ha, hc = moments(u, equations)
    a = ha / h
    c = hc / h
    v1, v2 = velocities(u, equations)

    c1 = c2 = equations.gravity * h
    for i in eachmoment(equations)
        # FIX: was `3 * a[i]` (linear); needs 3*a[i]^2 to match the Koellermeier-Pimentel-García
        # formula. Same bug as in the nonlinear file
        c1 += (3 * a[i]^2) / (2 * i + 1)
        c2 += (3 * c[i]^2) / (2 * i + 1)
    end
    c1 = sqrt(c1)
    c2 = sqrt(c2)

    return abs(v1) + c1, abs(v2) + c2
end

@inline function Trixi.max_abs_speed(u_ll,
                                     u_rr,
                                     orientation::Integer,
                                     equations::ShallowWaterLinearizedMomentEquations2D)
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)

    if orientation == 1
        m_ll = moments1(u_ll, equations)
        m_rr = moments1(u_rr, equations)
        v_ll = velocity1(u_ll, equations)
        v_rr = velocity1(u_rr, equations)
    else
        m_ll = moments2(u_ll, equations)
        m_rr = moments2(u_rr, equations)
        v_ll = velocity2(u_ll, equations)
        v_rr = velocity2(u_rr, equations)
    end

    c_ll = equations.gravity * h_ll
    c_rr = equations.gravity * h_rr
    for i in eachmoment(equations)
        # FIX: same squared-moment bug as max_abs_speeds above.
        c_ll += (3 * (m_ll[i] / h_ll)^2) / (2 * i + 1)
        c_rr += (3 * (m_rr[i] / h_rr)^2) / (2 * i + 1)
    end
    c_ll = sqrt(c_ll)
    c_rr = sqrt(c_rr)

    return max(abs(v_ll) + c_ll, abs(v_rr) + c_rr)
end

@inline function Trixi.cons2prim(u, equations::ShallowWaterLinearizedMomentEquations2D)
    h = waterheight(u, equations)
    ha, hc = moments(u, equations)
    b = u[end]

    H = h + b
    a = ha / h
    c = hc / h
    v1, v2 = velocities(u, equations)

    return SVector{2 * nmoments(equations) + 4, real(equations)}(H, v1, v2, a..., c..., b)
end

@inline function Trixi.prim2cons(prim, equations::ShallowWaterLinearizedMomentEquations2D)
    H = waterheight(prim, equations)
    v1 = prim[2]
    v2 = prim[3]
    a, c = moments(prim, equations)
    b = prim[end]

    h = H - b
    ha = h * a
    hc = h * c
    hv1 = h * v1
    hv2 = h * v2

    return SVector{2 * nmoments(equations) + 4, real(equations)}(h, hv1, hv2, ha..., hc..., b)
end

@inline function Trixi.cons2entropy(u, equations::ShallowWaterLinearizedMomentEquations2D)
    h = waterheight(u, equations)
    v1, v2 = velocities(u, equations)
    ha, hc = moments(u, equations)
    a = ha / h
    c = hc / h
    b = u[end]
    g = equations.gravity

    w1 = g * (h + b) - 0.5 * v1^2 - 0.5 * v2^2
    for i in eachmoment(equations)
        w1 -= 0.5 * a[i]^2 / (2 * i + 1)
        w1 -= 0.5 * c[i]^2 / (2 * i + 1)
    end

    w2 = v1
    w3 = v2

    w_moments1 = MVector{nmoments(equations), real(equations)}(undef)
    w_moments2 = MVector{nmoments(equations), real(equations)}(undef)

    for i in eachmoment(equations)
        w_moments1[i] = a[i] / (2 * i + 1)
        w_moments2[i] = c[i] / (2 * i + 1)
    end

    return SVector(w1, w2, w3, w_moments1..., w_moments2..., b)
end

@inline function Trixi.waterheight(u, equations::ShallowWaterLinearizedMomentEquations2D)
    return u[1]
end

@inline function moments1(u, equations::ShallowWaterLinearizedMomentEquations2D)
    return SVector{nmoments(equations), real(equations)}(u[i]
                                                         for i in (4:(nmoments(equations) + 3)))
end

@inline function moments2(u, equations::ShallowWaterLinearizedMomentEquations2D)
    return SVector{nmoments(equations), real(equations)}(u[i]
                                                         for i in ((nmoments(equations) + 4):(2 * nmoments(equations) + 3)))
end

# RS: no separate moments tuple wrapper here, as the one defined in 
# shallow_water_moments_2d.jl via Union already covers this type, since its body 
# just forwards to moments1/moments2, which already dispatch correctly.

@inline function velocity1(u, equations::ShallowWaterLinearizedMomentEquations2D)
    h = waterheight(u, equations)
    hv1 = u[2]

    return hv1 / h
end

@inline function velocity2(u, equations::ShallowWaterLinearizedMomentEquations2D)
    h = waterheight(u, equations)
    hv2 = u[3]

    return hv2 / h
end

@inline function Trixi.entropy(u, equations::ShallowWaterLinearizedMomentEquations2D)
    h = waterheight(u, equations)
    v1, v2 = velocities(u, equations)
    ha, hc = moments(u, equations)
    a = ha ./ h
    c = hc ./ h
    b = u[end]
    g = equations.gravity

    e = 0.5 * (h * v1^2 + h * v2^2 + g * h^2) + g * h * b

    for i in eachmoment(equations)
        e += 0.5 * h * a[i]^2 / (2 * i + 1)
        e += 0.5 * h * c[i]^2 / (2 * i + 1)
    end

    return e
end

@inline function Trixi.lake_at_rest_error(u, equations::ShallowWaterLinearizedMomentEquations2D)
    h = waterheight(u, equations)
    b = u[end]

    return abs(equations.H0 - (h + b))
end
end # @muladd