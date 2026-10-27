# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

####################################################################################################
# 2D Shallow Water Moment Equations
#####################################################################################################

@doc raw"""
    ShallowWaterMomentEquations2D(; gravity, H0 = zero(gravity), n_moments,
                                    nu = 0.1, lambda = 0.1, rho = 1000.0,
                                    nman = 0.0165)
Shallow water moment equations in two spatial dimensions. The equations are given by
```math
\begin{cases}
\partial_t h + \partial_x hv1 + \partial_y hv2 = 0, \\
\partial_t hv1 + \partial_x \left(hv1^2 + h\sum\limits_{i=1}^N \frac{\alpha_i^2}{2i+1}\right) + \partial_y \left(hv1v2 + h\sum\limits_{i=1}^N \frac{\alpha_i\beta_i}{2i+1}\right) = -gh\partial_x(h+b),\\
\partial_t hv2 + \partial_x \left(hv1v2 + h\sum\limits_{i=1}^N \frac{\alpha_i\beta_i}{2i+1}\right) + \partial_y \left(hv2^2 + h\sum\limits_{i=1}^N \frac{\beta_i^2}{2i+1}\right) = -gh\partial_y(h+b),\\
\partial_t h\alpha_i + \partial_x \left(2hv1\alpha_i + h\sum\limits_{j,k=1}^N A_{ijk}\alpha_j \alpha_k \right) + \partial_y \left(hv1\beta_i + hv2\alpha_i + h\sum\limits_{j,k=1}^N A_{ijk}\alpha_j \beta_k \right) =
                       v1(\partial_x h\alpha_i + \partial_y h\beta_i) - h\sum\limits_{j,k=1}^N B_{ijk} \alpha_k (\partial_x h\alpha_j + \partial_y h\beta_j),\\
\partial_t h\beta_i + \partial_x \left(hv1\beta_i + hv2\alpha_i + h\sum\limits_{j,k=1}^N A_{ijk}\beta_j \alpha_k \right) + \partial_y \left(2hv2\beta_i + h\sum\limits_{j,k=1}^N A_{ijk}\beta_j \beta_k \right) =
                       v2(\partial_x h\alpha_i + \partial_y h\beta_i) - h\sum\limits_{j,k=1}^N B_{ijk} \beta_k (\partial_x h\alpha_j + \partial_y h\beta_j),
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

#Use the notation (h, v1, v2, a1, ..., an_moments, c1, ...., cn_moments, b) for the primitive 
#variables and analog for the conservative ones

struct ShallowWaterMomentEquations2D{NVARS, NMOMENTS, RealT <: Real,
                                     Array3 <: AbstractArray{RealT, 3},
                                     Array2 <: AbstractArray{RealT, 2}} <:
       AbstractMomentEquations{2, NVARS, NMOMENTS} 
    gravity::RealT   # gravitational acceleration
    H0::RealT        # constant "lake-at-rest" total water height
    n_moments::Integer  # number of moments (per direction)
    # Moment matrices
    A::Array3
    B::Array3
    C::Array2
    # Friction related quantities
    nu::RealT       # kinematic viscosity
    lambda::RealT  # slip length
    rho::RealT    # fluid density (relevant for Manning friction)
    nman::RealT   # Manning roughness coefficient

    function ShallowWaterMomentEquations2D{NVARS, NMOMENTS, RealT, Array3, Array2}(gravity::RealT,
                                                                                    H0::RealT,
                                                                                    n_moments::Integer,
                                                                                    A::Array3,
                                                                                    B::Array3,
                                                                                    C::Array2,
                                                                                    nu::RealT,
                                                                                    lambda::RealT,
                                                                                    rho::RealT,
                                                                                    nman::RealT) where {
                                                                                                        NVARS,
                                                                                                        NMOMENTS,
                                                                                                        RealT <:
                                                                                                        Real,
                                                                                                        Array3 <:
                                                                                                        AbstractArray{RealT,
                                                                                                                      3},
                                                                                                        Array2 <:
                                                                                                        AbstractArray{RealT,
                                                                                                                      2}                                                                                                        
                                                                                                        }
        new(gravity, H0, n_moments, A, B, C, nu, lambda, rho, nman)
    end
end

# Allow for flexibility to set the gravitational acceleration and number of moments within an elixir
# depending on the application. Here `gravity=1.0` or `gravity=9.81` are common values for the
# gravitational acceleration. The reference total water height H0 defaults to 0.0 but is used for 
# the "lake-at-rest" well-balancedness test cases.
function ShallowWaterMomentEquations2D(; gravity, H0 = zero(gravity), n_moments,  
                                       nu = 0.1, lambda = 0.1, rho = 1000.0,
                                       nman = 0.0165)
    RealT = promote_type(typeof(gravity), typeof(H0))

    # Extract number of moments and variables
    NMOMENTS = n_moments    #n_moments/NMOMENTS is the number of basis functions (moments per dimension) 
    NVARS =  2 * NMOMENTS + 4    # (h, hv1, hv2, a1, ..., an_moments, c1, ..., cn_moments, b) #RP: changed number of varibales 

    # Compute moment matrices
    A = compute_A_tensor(n_moments)
    B = compute_B_tensor(n_moments)
    C = compute_C_matrix(n_moments)
    Array3 = promote_type(typeof(A), typeof(B))
    Array2 = typeof(C)
    return ShallowWaterMomentEquations2D{NVARS, NMOMENTS, RealT, Array3, Array2}(gravity,   
                                                                 H0,
                                                                 n_moments,
                                                                 A,
                                                                 B,
                                                                 C,
                                                                 nu,
                                                                 lambda,
                                                                 rho,
                                                                 nman)
end

@inline function Base.real(::ShallowWaterMomentEquations2D{NVARS, NMOMENTS, RealT}) where {
                                                                                           NVARS,
                                                                                           NMOMENTS,
                                                                                           RealT <:
                                                                                           Real
                                                                                           }
    RealT
end

Trixi.have_nonconservative_terms(::ShallowWaterMomentEquations2D) = True() 

function Trixi.varnames(::typeof(cons2cons), equations::ShallowWaterMomentEquations2D)  
    conservative_moments_a = ntuple(n -> "ha" * string(n), Val(nmoments(equations))) 
    conservative_moments_c = ntuple(n -> "hc" * string(n), Val(nmoments(equations)))

    return ("h", "hv1", "hv2", conservative_moments_a..., conservative_moments_c..., "b") 
end

# We use the total layer heights, H = ∑h + b as primitive variables for easier visualization and setting initial
# conditions
function Trixi.varnames(::typeof(cons2prim), equations::ShallowWaterMomentEquations2D) 
    primitive_moments_a = ntuple(n -> "a" * string(n), Val(nmoments(equations)))  
    primitive_moments_c = ntuple(n -> "c" * string(n), Val(nmoments(equations)))
    return ("H", "v1", "v2", primitive_moments_a..., primitive_moments_c..., "b")    
end

# Source term for friction
# """
#     source_term_bottom_friction(u, x, t, equations::ShallowWaterMomentEquations2D)

# For details see the paper:
# - Julian Koellermeier, Ernesto Pimentel-García (2022)
#   Steady states and well-balanced schemes for shallow water moment equations with topography
#   [DOI: 10.1016/j.amc.2022.127166](https://doi.org/10.1016/j.amc.2022.127166)
# """
@inline function TrixiShallowWater.source_term_bottom_friction(u, x, t,
                                                               equations::Union{ShallowWaterMomentEquations2D,
                                                                                ShallowWaterLinearizedMomentEquations2D}) 
    # Get waterheight, velocity and moments
    h = waterheight(u, equations)
    v1, v2 = velocities(u, equations)
    ha, hc = moments(u, equations) 
    a = ha / h
    c = hc / h

    sum_a = sum(a)
    sum_c = sum(c) 

    friction_v1 = -equations.nu / equations.lambda * (v1 + sum_a)
    friction_v2 = -equations.nu / equations.lambda * (v2 + sum_c)  

    friction_mom1 = MVector{nmoments(equations), real(equations)}(undef)
    friction_mom2 = MVector{nmoments(equations), real(equations)}(undef)   

    for i in eachmoment(equations)
        friction_mom1[i] = -(2 * i + 1) * equations.nu / equations.lambda * (v1 + sum_a)
        friction_mom2[i] = -(2 * i + 1) * equations.nu / equations.lambda * (v2 + sum_c)   
        for j in eachmoment(equations)
            friction_mom1[i] += -equations.nu / h * equations.C[i, j] * a[j]
            friction_mom2[i] += -equations.nu / h * equations.C[i, j] * c[j]   
        end
    end

    return SVector(0, friction_v1, friction_v2, friction_mom1..., friction_mom2..., 0)  
end

# Introduce Manning friction source term
@inline function source_term_manning_friction(u, x, t,
                                              equations::Union{ShallowWaterMomentEquations2D,
                                                               ShallowWaterLinearizedMomentEquations2D})    
    # Get waterheight, velocity and moments
    h = waterheight(u, equations)
    v1, v2 = velocities(u, equations)
    ha, hc = moments(u, equations) 
    a = ha / h
    c = hc / h 

    sum_a = sum(a)
    sum_c = sum(c) 

    friction_v1 = -equations.rho * equations.gravity * ((equations.nman^2) / h^(1 / 3)) *
                 (v1 + sum_a) * abs(v1 + sum_a)
    friction_v2 = -equations.rho * equations.gravity * ((equations.nman^2) / h^(1 / 3)) *   
                 (v2 + sum_c) * abs(v2 + sum_c)

    friction_mom1 = MVector{nmoments(equations), real(equations)}(undef)
    friction_mom2 = MVector{nmoments(equations), real(equations)}(undef)

    for i in eachmoment(equations)
        friction_mom1[i] = -(2 * i + 1) * equations.rho * equations.gravity *
                          ((equations.nman^2) / h^(1 / 3)) * (v1 + sum_a) *
                          abs(v1 + sum_a)
        friction_mom2[i] = -(2 * i + 1) * equations.rho * equations.gravity *
                          ((equations.nman^2) / h^(1 / 3)) * (v2 + sum_c) *
                          abs(v2 + sum_c)
        for j in eachmoment(equations)
            friction_mom1[i] += -equations.nu / h * equations.C[i, j] * a[j]
            friction_mom2[i] += -equations.nu / h * equations.C[i, j] * c[j]
        end
    end

    return SVector(0, friction_v1, friction_v2, friction_mom1..., friction_mom2..., 0)  
end

"""
    boundary_condition_slip_wall(u_inner, orientation_or_normal, x, t, surface_flux_function,
                                 equations::ShallowWaterMomentEquations2D)

Create a boundary state by reflecting the normal velocity component and keep
the tangential velocity component unchanged. The boundary water height is taken from
the internal value.

For details see Section 9.2.5 of the book:
- Eleuterio F. Toro (2001)
  Shock-Capturing Methods for Free-Surface Shallow Flows
  1st edition
  ISBN 0471987662
"""
@inline function Trixi.boundary_condition_slip_wall(u_inner,
                                                    orientation::Integer, 
                                                    direction,
                                                    x,
                                                    t,
                                                    surface_flux_functions,
                                                    equations::ShallowWaterMomentEquations2D)
    surface_flux_function, nonconservative_flux_function = surface_flux_functions

    # Create the "external" boundary solution state
    h = u_inner[1]
    hv1 = u_inner[2]
    hv2 = u_inner[3] 
    ha, hc = moments(u_inner, equations) 
    b = u_inner[end]

    if orientation == 1
    # wall normal is the x-direction: flip hv1 and every a-moment, keep hv2/c-moments
        u_boundary = SVector(h, -hv1, hv2, (-ha)..., hc..., b)
    else
    # wall normal is the y-direction: flip hv2 and every c-moment, keep hv1/a-moments
        u_boundary = SVector(h, hv1, -hv2, ha..., (-hc)..., b)
    end

    # Calculate the boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation,
                                     equations)
        noncons_flux = nonconservative_flux_function(u_inner,
                                                     u_boundary,
                                                     orientation,
                                                     equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation,
                                     equations)
        noncons_flux = nonconservative_flux_function(u_inner,
                                                     u_boundary,
                                                     orientation,
                                                     equations)
    end
    return flux, noncons_flux
end

# Calculate 2D advective portion of the flux for a single point
@inline function Trixi.flux(u,
                            orientation::Integer,
                            equations::ShallowWaterMomentEquations2D)   
    # Extract conservative variables    
    h = waterheight(u, equations)
    hv1 = u[2]
    hv2 = u[3]  
    ha, hc = moments(u, equations) 

    # Compute primitive variables
    v1 = velocity1(u, equations)
    v2 = velocity2(u, equations)
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
        
        f_moments1 = MVector{nmoments(equations), real(equations)}(ha * v1 + hv1 * a)   
        f_moments2 = MVector{nmoments(equations), real(equations)}(hc * v1 + ha * v2)
   
        for i in eachmoment(equations),
            j in eachmoment(equations),
            k in eachmoment(equations)


            f_moments1[i] += equations.A[i, j, k] * ha[j] * a[k]
            f_moments2[i] += equations.A[i, j, k] * ha[j] * c[k]

        end
    else  
        f1 = hv2
        f2 = hv1 * v2
        f3 = hv2 * v2
        for i in eachmoment(equations)
            f2 += a[i] * hc[i] / (2 * i + 1)
            f3 += hc[i] * c[i] / (2 * i + 1)
        end

        f_moments1 = MVector{nmoments(equations), real(equations)}(hc * v1 + ha * v2)
        f_moments2 = MVector{nmoments(equations), real(equations)}(hc * v2 + hv2 * c)  

        for i in eachmoment(equations),
            j in eachmoment(equations),
            k in eachmoment(equations)
            
        # FIX: was `c[j] * a[k]` / `c[j] * c[k]`: missing an h factor (same issue as the
        # x-direction block above) AND the wrong variables: matching flux_ec's
        # verified y-direction pattern (ha_avg[j]*c_avg[k], hc_avg[j]*c_avg[k]), f_moments1
        # here needs ha (not c) as its conservative factor
            f_moments1[i] += equations.A[i, j, k] * ha[j] * c[k]
            f_moments2[i] += equations.A[i, j, k] * hc[j] * c[k]

        end
    end

    return SVector{2 * nmoments(equations) + 4, real(equations)}(f1, f2, f3, f_moments1..., f_moments2..., 0)   
end

"""
    flux_ec(u_ll, u_rr, orientation::Integer,
                                     equations::ShallowWaterMomentEquations1D)

Total energy conservative split form, without the hydrostatic pressure.
When the bottom topography is nonzero this scheme will be well-balanced when used with the 
nonconservative [`flux_nonconservative_ec`](@ref).

To obtain an entropy stable formulation the `surface_flux` can be set as
`FluxPlusDissipation(flux_ec, DissipationLocalLaxFriedrichs()), flux_nonconservative_ec`.
"""
@inline function flux_ec(u_ll,
                         u_rr,
                         orientation::Integer,
                         equations::ShallowWaterMomentEquations2D)  #
    # Unpack left and right state
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    hv1_ll = u_ll[2]
    hv1_rr = u_rr[2]
    hv2_ll = u_ll[3]
    hv2_rr = u_rr[3] # RS- FIX: was `hv3_rr = u_rr[4]` 
    ha_ll, hc_ll = moments(u_ll, equations)
    ha_rr, hc_rr = moments(u_rr, equations)   

    # Get the velocities and primitive moments on either side
    v1_ll = velocity1(u_ll, equations)
    v1_rr = velocity1(u_rr, equations)
    v2_ll = velocity2(u_ll, equations)
    v2_rr = velocity2(u_rr, equations)
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

    # Compute the flux components
    if orientation == 1 
        f1 = hv1_avg
        f2 = hv1_avg * v1_avg
        f3 = hv1_avg * v2_avg 
        for i in eachmoment(equations)
            f2 += ha_avg[i] * a_avg[i] / (2 * i + 1)
            f3 += ha_avg[i] * c_avg[i] / (2 * i + 1)
        end

        # Compute the moment fluxes.
        f_moments1 = MVector{nmoments(equations), real(equations)}(hv1_avg * a_avg +
                                                                ha_avg * v1_avg)
        f_moments2 = MVector{nmoments(equations), real(equations)}(hv1_avg * c_avg +
                                                                ha_avg * v2_avg)   

        for i in eachmoment(equations),
            j in eachmoment(equations),
            k in eachmoment(equations)

            f_moments1[i] += equations.A[i, j, k] * ha_avg[j] * a_avg[k]
            f_moments2[i] += equations.A[i, j, k] * ha_avg[j] * c_avg[k]
        end
    else    
        f1 = hv2_avg
        f2 = hv1_avg * v2_avg
        f3 = hv2_avg * v2_avg
        for i in eachmoment(equations)
            f2 += a_avg[i] * hc_avg[i] / (2 * i + 1)
            f3 += hc_avg[i] * c_avg[i] / (2 * i + 1)
        end

        # Compute the moment fluxes.
        f_moments1 = MVector{nmoments(equations), real(equations)}(v1_avg * hc_avg +
                                                                a_avg * hv2_avg)
        f_moments2 = MVector{nmoments(equations), real(equations)}(hv2_avg * c_avg +
                                                                hc_avg * v2_avg)   

        for i in eachmoment(equations),
            j in eachmoment(equations),
            k in eachmoment(equations)

            f_moments1[i] += equations.A[i, j, k] * ha_avg[j] * c_avg[k]
            f_moments2[i] += equations.A[i, j, k] * hc_avg[j] * c_avg[k]
        end
    end

    return SVector{2 * nmoments(equations) + 4, real(equations)}(f1, f2, f3, f_moments1..., f_moments2..., 0)  
end

"""
    flux_nonconservative_ec(u_ll, u_rr, orientation::Integer,
                                     equations::ShallowWaterTwoLayerEquations1D)

Non-symmetric path-conservative two-point flux discretizing the nonconservative (source) term
that contains the gradients of the bottom topography and waterheights from the coupling between layers
and the nonconservative pressure formulation [`ShallowWaterMomentEquations1D`](@ref).

When the bottom topography is nonzero this scheme will be well-balanced when used with [`flux_ec`](@ref).
"""
@inline function flux_nonconservative_ec(u_ll,
                                         u_rr,
                                         orientation::Integer,
                                         equations::ShallowWaterMomentEquations2D)  #RP: change type of equations
    # Unpack left and right state
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    ha_ll, hc_ll = moments(u_ll, equations)
    ha_rr, hc_rr = moments(u_rr, equations)
    a_ll = ha_ll / h_ll
    c_ll = hc_ll / h_ll 
    b_ll = u_ll[end]
    b_rr = u_rr[end]

    # Get the velocities and primitive moments on either side
    v1_ll = velocity1(u_ll, equations)
    v2_ll = velocity2(u_ll, equations) 

    # Compute the jumps
    h_jump = h_rr - h_ll
    b_jump = b_rr - b_ll
    ha_jump = ha_rr - ha_ll
    hc_jump = hc_rr - hc_ll

    # Get the gravitational acceleration
    g = equations.gravity

    if orientation == 1
        # Compute nonconservative flux components
        f2 = g * h_ll * (h_jump + b_jump)
        f3 = zero(g)   # or: 0.0
        # Compute the moment fluxes.
        f_moments1 = MVector{nmoments(equations), real(equations)}(-v1_ll * ha_jump)    
        f_moments2 = MVector{nmoments(equations), real(equations)}(-v2_ll * ha_jump)

        for i in eachmoment(equations),
            j in eachmoment(equations),
            k in eachmoment(equations)

            f_moments1[i] += equations.B[i, j, k] * a_ll[k] * ha_jump[j]
            f_moments2[i] += equations.B[i, j, k] * c_ll[k] * ha_jump[j]
        end
    else
         
        # Compute nonconservative flux components
        f2 = zero(g)   # or: 0.0
        f3 = g * h_ll * (h_jump + b_jump)
        # Compute the moment fluxes.
        f_moments1 = MVector{nmoments(equations), real(equations)}(-v1_ll * hc_jump)    
        f_moments2 = MVector{nmoments(equations), real(equations)}(-v2_ll * hc_jump)

        for i in eachmoment(equations),
            j in eachmoment(equations),
            k in eachmoment(equations)

            f_moments1[i] += equations.B[i, j, k] * a_ll[k] * hc_jump[j]
            f_moments2[i] += equations.B[i, j, k] * c_ll[k] * hc_jump[j]
        end
    end

    return SVector{2 * nmoments(equations) + 4, real(equations)}(0, f2, f3, f_moments1..., f_moments2...,0)   
end

# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the bottom
# topography. For nonzero bottom topography [`Trixi.DissipationLaxFriedrichsEntropyVariables`](@extref)
# should be used instead.
# This one is dimension-agnostic (it only drops the last, bottom-topography
# entry of the jump) so no 2D-specific changes are needed here aside from the type of the equations.
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll,
                                                              u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::ShallowWaterMomentEquations2D)
    λ = dissipation.max_abs_speed(u_ll,
                                  u_rr,
                                  orientation_or_normal_direction,
                                  equations) 
    diss = -0.5 * λ * (u_rr - u_ll)
    return SVector(@views diss[1:(end - 1)]..., zero(eltype(u_ll)))
end

# Specialized [`Trixi.DissipationLaxFriedrichsEntropyVariables`](@extref) for the SWME that avoids spurious 
# dissipation in the bottom topography
@inline function (dissipation::DissipationLaxFriedrichsEntropyVariables)(u_ll,
                                                                         u_rr,
                                                                         orientation_or_normal_direction,
                                                                         equations::Union{ShallowWaterLinearizedMomentEquations2D,
                                                                                          ShallowWaterMomentEquations2D})
    λ = dissipation.max_abs_speed(u_ll,
                                  u_rr,
                                  orientation_or_normal_direction,
                                  equations)

    # Convert to entropy variables
    w_ll = Trixi.cons2entropy(u_ll, equations)
    w_rr = Trixi.cons2entropy(u_rr, equations)

    Nred = 2 * nmoments(equations) + 3  
    # Exclude the bottom topography from the jump
    w_jump = SVector{Nred}(@views (w_rr - w_ll)[1:(end - 1)])

    # Compute the matrix H = du/dw at the average state
    h_avg = 0.5 * (u_ll[1] + u_rr[1])
    v1_avg = 0.5 * (velocity1(u_ll, equations) + velocity1(u_rr, equations))
    v2_avg = 0.5 * (velocity2(u_ll, equations) + velocity2(u_rr, equations))
    a_avg = 0.5 * (moments1(u_ll, equations) / u_ll[1] +  moments1(u_rr, equations) / u_rr[1])
    c_avg = 0.5 * (moments2(u_ll, equations) / u_ll[1] + moments2(u_rr, equations) / u_rr[1])

    g = equations.gravity

    # Construct the H matrix from H = 1/g * (y y' + diag(z))
    y = SVector{Nred, real(equations)}(1, v1_avg, v2_avg, a_avg..., c_avg...)  
    
    z = zero(MVector{Nred, real(equations)})
    z[2] = g * h_avg 
    z[3] = g * h_avg
    n = nmoments(equations)  
    for i in 1:n
        z[3 + i] = (2i + 1) * g * h_avg
        z[3 + n + i] = (2i + 1) * g * h_avg
    end

    diss = SVector{Nred, real(equations)}(-0.5 * λ / g *
                                          (y * (y' * w_jump) + z .* w_jump))

    return SVector{Nred + 1, real(equations)}(diss..., zero(eltype(u_ll)))
end

# The eigenvalues are approximate with those of the SWLME taken from:
# - Julian Koellermeier, Ernesto Pimentel-García (2022)
#   Steady states and well-balanced schemes for shallow water moment equations with topography
#   [DOI: 10.1016/j.amc.2022.127166](https://doi.org/10.1016/j.amc.2022.127166)
#   The paper considers the 1D-case only, we need a reference for the 2D eigenvalues. A proof of hyperbolicity for the linearized shallow water moment model under special assumptions can be found in:
# - Matthew Bauerle, Andrew J. CHristlieb, Mingchang Ding and Juntao Huang (2024)
#   On the rotational invariance and hyperbolicity of shallow water moment equations in two dimensions
#   [DOI: https://doi.org/10.48550/arXiv.2306.07202]
#   Linearized equations are rotational invariant, take the maximal absolut value of both directions as an estimator

@inline function Trixi.max_abs_speeds(u, equations::ShallowWaterMomentEquations2D)
    h = waterheight(u, equations)
    ha, hc = moments(u, equations)
    v1, v2 = velocities(u, equations) 

    # calculate the wave celerity
    c1 = c2 = max(0.0, equations.gravity * h)
    for i in eachmoment(equations)

        c1 += (3 * (ha[i] / h)^2) / (2 * i + 1)
        c2 += (3 * (hc[i] / h)^2) / (2 * i + 1)
    end
    c1 = sqrt(c1)
    c2 = sqrt(c2)


    return (abs(v1) + c1),(abs(v2) + c2)
end

# Less "cautious", i.e., less overestimating `λ_max` compared to `max_abs_speed_naive`
# Used at interfaces for a given `orientation`, so (unlike `max_abs_speeds` above) this picks
# the a/v1 or c/v2 block depending on direction.
@inline function Trixi.max_abs_speed(u_ll,
                                     u_rr,
                                     orientation::Integer,
                                     equations::ShallowWaterMomentEquations2D)
    # Unpack left and right state
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)

    if orientation == 1
        ha_ll = moments1(u_ll, equations)
        ha_rr = moments1(u_rr, equations)
        v_ll = velocity1(u_ll, equations)
        v_rr = velocity1(u_rr, equations)
    else
        ha_ll = moments2(u_ll, equations)
        ha_rr = moments2(u_rr, equations)
        v_ll = velocity2(u_ll, equations)
        v_rr = velocity2(u_rr, equations)
    end


    # Calculate the wave celerity on the left and right
    c_ll = max(0.0, equations.gravity * h_ll)
    c_rr = max(0.0, equations.gravity * h_rr)
    for i in eachmoment(equations)
        c_ll += (3 * (ha_ll[i] / h_ll)^2) / (2 * i + 1) 
        c_rr += (3 * (ha_rr[i] / h_rr)^2) / (2 * i + 1)
    end
    c_ll = sqrt(c_ll)
    c_rr = sqrt(c_rr)

    return (max(abs(v_ll) + c_ll, abs(v_rr) + c_rr))
end

# Convert conservative variables to primitive
@inline function Trixi.cons2prim(u, equations::ShallowWaterMomentEquations2D)   
    # Extract conservative variables
    h = waterheight(u, equations)
    ha, hc = moments(u, equations)   
    b = u[end]

    # Compute the total water height, velocity and primitive moments
    H = h + b
    a = ha / h
    c = hc / h
    v1 = velocity1(u, equations)
    v2 = velocity2(u, equations)    

    return SVector{2 * nmoments(equations) + 4, real(equations)}(H, v1, v2, a..., c..., b)  
end

# Convert primitive to conservative variables
@inline function Trixi.prim2cons(prim, equations::ShallowWaterMomentEquations2D)   
    # To extract the total layer height and velocity we reuse the water height and momentum functions 
    # from the conservative variables.
    H = waterheight(prim, equations)
    v1 = prim[2]
    v2 = prim[3]    
    a, c = moments(prim, equations)     # For primitive variables this extracts the primitive moments

    b = prim[end]

    # Compute the conservative variables
    h = H - b
    ha = h * a
    hc = h * c
    hv1 = h * v1
    hv2 = h * v2

    return SVector{2 * nmoments(equations) + 4, real(equations)}(h, hv1, hv2, ha..., hc..., b)  
end

# Convert conservative variables to entropy variables
@inline function Trixi.cons2entropy(u, equations::ShallowWaterMomentEquations2D)   
    # Extract conservative variables and compute velocity
    h = waterheight(u, equations)
    v1, v2 = velocities(u, equations)
    ha, hc = moments(u, equations) 
    a = ha / h
    c = hc / h
    b = u[end]
    g = equations.gravity

    # Calculate entropy variables
    w1 = g * (h + b) - 0.5 * v1^2 - 0.5 * v2^2
    # add moment contributions
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

@inline function Trixi.waterheight(u, equations::ShallowWaterMomentEquations2D) 
    return u[1]
end

@inline function moments1(u, equations::ShallowWaterMomentEquations2D)  
    return SVector{nmoments(equations), real(equations)}(u[i]
                                                         for i in (4:(nmoments(equations) + 3)))
end

@inline function moments2(u, equations::ShallowWaterMomentEquations2D)
    return SVector{nmoments(equations), real(equations)}(u[i]
                                                         for i in ((nmoments(equations) + 4):(2 * nmoments(equations) + 3)))   
end

@inline function moments(u,
                         equations::Union{ShallowWaterMomentEquations2D,
                                          ShallowWaterLinearizedMomentEquations2D})
    return moments1(u, equations), moments2(u, equations)
end

@inline function velocity1(u, equations::ShallowWaterMomentEquations2D)
    h = waterheight(u, equations)
    hv1 = u[2]

    return hv1 / h
end

@inline function velocity2(u, equations::ShallowWaterMomentEquations2D)
    h = waterheight(u, equations)
    hv2 = u[3]

    return hv2 / h
end

@inline function velocities(u,
                            equations::Union{ShallowWaterMomentEquations2D,
                                             ShallowWaterLinearizedMomentEquations2D})
    return velocity1(u, equations), velocity2(u, equations)
end

# The entropy function for the linearized shallow water moment equations is the total energy
@inline function Trixi.entropy(u, equations::ShallowWaterMomentEquations2D) 
    h = waterheight(u, equations)
    v1 = velocity1(u, equations)
    v2 = velocity2(u, equations)   
    a = moments1(u, equations) ./ h
    c = moments2(u, equations) ./ h 
    b = u[end]
    g = equations.gravity

    # Calculate the total energy for the mean flow
    e = 0.5 * (h * v1^2 + v2^2 + g * h^2) + g * h * b  

    # add the contribution from moments
    for i in eachmoment(equations)
        e += 0.5 * h * a[i]^2 / (2 * i + 1)
        e += 0.5 * h * c[i]^2 / (2 * i + 1) 
    end

    return e
end

# Calculate the error for the "lake-at-rest" test case where H = h + b should
# be a constant value over time. 
# Note, assumes there is a single reference water height `H0` with which to compare.
@inline function Trixi.lake_at_rest_error(u, equations::ShallowWaterMomentEquations2D)  
    h = waterheight(u, equations)
    b = u[end]

    return abs(equations.H0 - (h + b))
end

# Entropy dissipation due to friction source terms
@inline function dwdP_Ns(u,
                         equations::Union{ShallowWaterMomentEquations2D,
                                          ShallowWaterLinearizedMomentEquations2D}) 
    w = cons2entropy(u, equations)
    P = source_term_bottom_friction(u, zero(real(equations)), zero(real(equations)),
                                    equations)
    return w' * P
end

@inline function dwdP_MM(u,
                         equations::Union{ShallowWaterMomentEquations2D,
                                          ShallowWaterLinearizedMomentEquations2D})   
    w = cons2entropy(u, equations)
    P = source_term_manning_friction(u, zero(real(equations)), zero(real(equations)),
                                     equations)
    return w' * P
end
end # @muladd
