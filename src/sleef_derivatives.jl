# Recipe: https://github.com/ThummeTo/ForwardDiffChainRules.jl
# https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/fastmath_able.jl

import ChainRulesCore
import SLEEFPirates
import DiffRules
import ForwardDiff
using IrrationalConstants: logtwo, logten, twoπ, sqrtπ, invsqrtπ
using RealDot: realdot

## sin
function ChainRulesCore.rrule(::typeof(SLEEFPirates.sin), x::Number)
    sinx, cosx = SLEEFPirates.sincos(x)
    sin_pullback(Δy) = (ChainRulesCore.NoTangent(), cosx' * Δy)
    return (sinx, sin_pullback)
end

function ChainRulesCore.frule((_, Δx), ::typeof(SLEEFPirates.sin), x::Number)
    sinx, cosx = SLEEFPirates.sincos(x)
    return (sinx, cosx * Δx)
end

function SLEEFPirates.sin(x::ForwardDiff.Dual{T}) where {T}
    sinx, cosx = SLEEFPirates.sincos(ForwardDiff.value(x))
    ForwardDiff.Dual{T}(sinx, cosx * ForwardDiff.partials(x))
end

## sin_fast
function ChainRulesCore.rrule(::typeof(SLEEFPirates.sin_fast), x::Number)
    sinx, cosx = SLEEFPirates.sincos_fast(x)
    sin_pullback(Δy) = (ChainRulesCore.NoTangent(), cosx' * Δy)
    return (sinx, sin_pullback)
end

function ChainRulesCore.frule((_, Δx), ::typeof(SLEEFPirates.sin_fast), x::Number)
    sinx, cosx = SLEEFPirates.sincos_fast(x)
    return (sinx, cosx * Δx)
end

function SLEEFPirates.sin_fast(x::ForwardDiff.Dual{T}) where {T}
    sinx, cosx = SLEEFPirates.sincos_fast(ForwardDiff.value(x))
    ForwardDiff.Dual{T}(sinx, cosx * ForwardDiff.partials(x))
end

## cos
function ChainRulesCore.rrule(::typeof(SLEEFPirates.cos), x::Number)
    sinx, cosx = SLEEFPirates.sincos(x)
    cos_pullback(Δy) = (ChainRulesCore.NoTangent(), -sinx' * Δy)
    return (cosx, cos_pullback)
end

function ChainRulesCore.frule((_, Δx), ::typeof(SLEEFPirates.cos), x::Number)
    sinx, cosx = sincos(x)
    return (cosx, -sinx * Δx)
end

function SLEEFPirates.cos(x::ForwardDiff.Dual{T}) where {T}
    sinx, cosx = SLEEFPirates.sincos(ForwardDiff.value(x))
    ForwardDiff.Dual{T}(cosx, -sinx * ForwardDiff.partials(x))
end

## cos_fast
function ChainRulesCore.rrule(::typeof(SLEEFPirates.cos_fast), x::Number)
    sinx, cosx = SLEEFPirates.sincos_fast(x)
    cos_pullback(Δy) = (ChainRulesCore.NoTangent(), -sinx' * Δy)
    return (cosx, cos_pullback)
end

function ChainRulesCore.frule((_, Δx), ::typeof(SLEEFPirates.cos_fast), x::Number)
    sinx, cosx = SLEEFPirates.sincos_fast(x)
    return (cosx, -sinx * Δx)
end

function SLEEFPirates.cos_fast(x::ForwardDiff.Dual{T}) where {T}
    sinx, cosx = SLEEFPirates.sincos_fast(ForwardDiff.value(x))
    ForwardDiff.Dual{T}(cosx, -sinx * ForwardDiff.partials(x))
end

## tan & tan_fast
ChainRulesCore.@scalar_rule SLEEFPirates.tan(x) 1 + Ω^2
ChainRulesCore.@scalar_rule SLEEFPirates.tan_fast(x) 1 + Ω^2

DiffRules.@define_diffrule SLEEFPirates.tan(x) = :(1 + SLEEFPirates.tan($x)^2)
DiffRules.@define_diffrule SLEEFPirates.tan_fast(x) = :(1 + SLEEFPirates.tan_fast($x)^2)

eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :tan))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :tan_fast))

# cosh
function ChainRulesCore.rrule(::typeof(SLEEFPirates.cosh), x::Number)
    sinhx, coshx = SLEEFPirates.sincosh(x)
    cos_pullback(Δy) = (ChainRulesCore.NoTangent(), sinhx' * Δy)
    return (coshx, cos_pullback)
end

function ChainRulesCore.frule((_, Δx), ::typeof(SLEEFPirates.cosh), x::Number)
    sinhx, coshx = SLEEFPirates.sincosh(x)
    return (coshx, sinhx * Δx)
end

function SLEEFPirates.cosh(x::ForwardDiff.Dual{T}) where {T}
    sinhx, coshx = SLEEFPirates.sincosh(ForwardDiff.value(x))
    ForwardDiff.Dual{T}(coshx, sinhx * ForwardDiff.partials(x))
end

# sinh
function ChainRulesCore.rrule(::typeof(SLEEFPirates.sinh), x::Number)
    sinhx, coshx = SLEEFPirates.sincosh(x)
    sin_pullback(Δy) = (ChainRulesCore.NoTangent(), coshx' * Δy)
    return (sinhx, cos_pullback)
end

function ChainRulesCore.frule((_, Δx), ::typeof(SLEEFPirates.sinh), x::Number)
    sinhx, coshx = SLEEFPirates.sincosh(x)
    return (sinhx, coshx * Δx)
end

function SLEEFPirates.sinh(x::ForwardDiff.Dual{T}) where {T}
    sinhx, coshx = SLEEFPirates.sincosh(ForwardDiff.value(x))
    ForwardDiff.Dual{T}(sinhx, coshx * ForwardDiff.partials(x))
end

# tanh
ChainRulesCore.@scalar_rule SLEEFPirates.tanh(x) 1 - Ω^2

DiffRules.@define_diffrule SLEEFPirates.tanh(x) = :(1 - SLEEFPirates.tanh($x)^2)

eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :tanh))

# Trig- Inverses
ChainRulesCore.@scalar_rule SLEEFPirates.acos(x) -(inv(sqrt(1 - x^2)))
ChainRulesCore.@scalar_rule SLEEFPirates.asin(x) inv(sqrt(1 - x^2))
ChainRulesCore.@scalar_rule SLEEFPirates.atan(x) inv(1 + x^2)
ChainRulesCore.@scalar_rule SLEEFPirates.acos_fast(x) -(inv(sqrt(1 - x^2)))
ChainRulesCore.@scalar_rule SLEEFPirates.asin_fast(x) inv(sqrt(1 - x^2))
ChainRulesCore.@scalar_rule SLEEFPirates.atan_fast(x) inv(1 + x^2)

DiffRules.@define_diffrule SLEEFPirates.acos(x) = :(-(inv(sqrt(1 - $x^2))))
DiffRules.@define_diffrule SLEEFPirates.asin(x) = :(inv(sqrt(1 - $x^2)))
DiffRules.@define_diffrule SLEEFPirates.atan(x) = :(inv(1 + $x^2))
DiffRules.@define_diffrule SLEEFPirates.acos_fast(x) = :(-(inv(sqrt(1 - $x^2))))
DiffRules.@define_diffrule SLEEFPirates.asin_fast(x) = :(inv(sqrt(1 - $x^2)))
DiffRules.@define_diffrule SLEEFPirates.atan_fast(x) = :(inv(1 + $x^2))

eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :acos))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :asin))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :atan))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :acos_fast))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :asin_fast))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :atan_fast))

# Trig-Hyperbolic Inverses
ChainRulesCore.@scalar_rule SLEEFPirates.acosh(x) inv(sqrt(x^2 - 1))
ChainRulesCore.@scalar_rule SLEEFPirates.asinh(x) inv(sqrt(x^2 + 1))
ChainRulesCore.@scalar_rule SLEEFPirates.atanh(x) inv(1 - x^2)

DiffRules.@define_diffrule SLEEFPirates.acosh(x) = :(inv(sqrt($x^2 - 1)))
DiffRules.@define_diffrule SLEEFPirates.asinh(x) = :(inv(sqrt($x^2 + 1)))
DiffRules.@define_diffrule SLEEFPirates.atanh(x) = :(inv(1 - $x^2))

eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :acosh))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :asinh))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :atanh))

# Trig-Multivariate
ChainRulesCore.@scalar_rule SLEEFPirates.atan(y, x) @setup(u = x^2 + y^2) (x / u, -y / u)
ChainRulesCore.@scalar_rule SLEEFPirates.atan_fast(y, x) @setup(u = x^2 + y^2) (x / u, -y / u)

DiffRules.@define_diffrule SLEEFPirates.atan(x, y) = :($y / ($x^2 + $y^2)), :(-$x / ($x^2 + $y^2))
DiffRules.@define_diffrule SLEEFPirates.atan_fast(x, y) = :($y / ($x^2 + $y^2)), :(-$x / ($x^2 + $y^2))

eval(ForwardDiff.binary_dual_definition(:SLEEFPirates, :atan))
eval(ForwardDiff.binary_dual_definition(:SLEEFPirates, :atan_fast))

ChainRulesCore.@scalar_rule SLEEFPirates.sincos(x) @setup((sinx, cosx) = Ω) cosx -sinx

@inline function SLEEFPirates.sincos(d::ForwardDiff.Dual{T}) where {T}
    sd, cd = SLEEFPirates.sincos(ForwardDiff.value(d))
    return (ForwardDiff.Dual{T}(sd, cd * ForwardDiff.partials(d)), ForwardDiff.Dual{T}(cd, -sd * ForwardDiff.partials(d)))
end

# exponents
ChainRulesCore.@scalar_rule SLEEFPirates.cbrt(x) inv(3 * Ω^2)
ChainRulesCore.@scalar_rule SLEEFPirates.exp(x) Ω
ChainRulesCore.@scalar_rule SLEEFPirates.exp10(x) logten * Ω
ChainRulesCore.@scalar_rule SLEEFPirates.exp2(x) logtwo * Ω
ChainRulesCore.@scalar_rule SLEEFPirates.expm1(x) SLEEFPirates.exp(x)
ChainRulesCore.@scalar_rule SLEEFPirates.expm1_fast(x) SLEEFPirates.exp(x)
ChainRulesCore.@scalar_rule SLEEFPirates.log(x) inv(x)
ChainRulesCore.@scalar_rule SLEEFPirates.log_fast(x) inv(x)
ChainRulesCore.@scalar_rule SLEEFPirates.log10(x) inv(logten * x)
ChainRulesCore.@scalar_rule SLEEFPirates.log10_fast(x) inv(logten * x)
ChainRulesCore.@scalar_rule SLEEFPirates.log1p(x) inv(x + 1)
ChainRulesCore.@scalar_rule SLEEFPirates.log2(x) inv(logtwo * x)
ChainRulesCore.@scalar_rule SLEEFPirates.log2_fast(x) inv(logtwo * x)

DiffRules.@define_diffrule SLEEFPirates.cbrt(x) = :(inv(3 * SLEEFPirates.cbrt($x)^2))
DiffRules.@define_diffrule SLEEFPirates.exp(x) = :(SLEEFPirates.exp($x))
DiffRules.@define_diffrule SLEEFPirates.exp10(x) = :($logten * SLEEFPirates.exp10($x))
DiffRules.@define_diffrule SLEEFPirates.exp2(x) = :($logtwo * SLEEFPirates.exp2($x))
DiffRules.@define_diffrule SLEEFPirates.expm1(x) = :(SLEEFPirates.exp($x))
DiffRules.@define_diffrule SLEEFPirates.expm1_fast(x) = :(SLEEFPirates.exp($x))
DiffRules.@define_diffrule SLEEFPirates.log(x) = :(inv($x))
DiffRules.@define_diffrule SLEEFPirates.log_fast(x) = :(inv($x))
DiffRules.@define_diffrule SLEEFPirates.log10(x) = :(inv($logten * $x))
DiffRules.@define_diffrule SLEEFPirates.log10_fast(x) = :(inv($logten * $x))
DiffRules.@define_diffrule SLEEFPirates.log1p(x) = :(inv($x + 1))
DiffRules.@define_diffrule SLEEFPirates.log2(x) = :(inv($logtwo * $x))
DiffRules.@define_diffrule SLEEFPirates.log2_fast(x) = :(inv($logtwo * $x))

eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :cbrt))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :exp))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :exp10))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :exp2))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :expm1))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :expm1_fast))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :log))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :log_fast))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :log10))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :log10_fast))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :log1p))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :log2))
eval(ForwardDiff.unary_dual_definition(:SLEEFPirates, :log2_fast))

# Hypot
function ChainRulesCore.frule((_, Δx, Δy), ::typeof(SLEEFPirates.hypot), x::T, y::T) where {T<:Union{Real,Complex}}
    Ω = SLEEFPirates.hypot(x, y)
    n = ifelse(iszero(Ω), one(Ω), Ω)
    ∂Ω = (realdot(x, Δx) + realdot(y, Δy)) / n
    return Ω, ∂Ω
end

function rrule(::typeof(SLEEFPirates.hypot), x::T, y::T) where {T<:Union{Real,Complex}}
    Ω = SLEEFPirates.hypot(x, y)
    function hypot_pullback(ΔΩ)
        c = real(ΔΩ) / ifelse(iszero(Ω), one(Ω), Ω)
        return (NoTangent(), c * x, c * y)
    end
    return (Ω, hypot_pullback)
end

DiffRules.@define_diffrule SLEEFPirates.hypot(x, y) = :($x / SLEEFPirates.hypot($x, $y)), :($y / SLEEFPirates.hypot($x, $y))

eval(ForwardDiff.binary_dual_definition(:SLEEFPirates, :hypot))

# pow
DiffRules.@define_diffrule SLEEFPirates.pow(x, y) = :($y * SLEEFPirates.pow($x, ($y - 1))), :(SLEEFPirates.pow($x, $y) * SLEEFPirates.log($x))
DiffRules.@define_diffrule SLEEFPirates.pow_fast(x, y) = :($y * SLEEFPirates.pow_fast($x, ($y - 1))), :(SLEEFPirates.pow_fast($x, $y) * SLEEFPirates.log_fast($x))

for (f,g) in ((:(SLEEFPirates.pow), :(SLEEFPirates.log)), (:(SLEEFPirates.pow_fast), :(SLEEFPirates.log_fast)))
    @eval begin
        ForwardDiff.@define_binary_dual_op(
            $f,
            begin
                vx, vy = ForwardDiff.value(x), ForwardDiff.value(y)
                expv = ($f)(vx, vy)
                powval = vy * ($f)(vx, vy - 1)
                if ForwardDiff.isconstant(y)
                    logval = one(expv)
                elseif iszero(vx) && vy > 0
                    logval = zero(vx)
                else
                    logval = expv * $g(vx)
                end
                new_partials = ForwardDiff._mul_partials(ForwardDiff.partials(x), ForwardDiff.partials(y), powval, logval)
                return ForwardDiff.Dual{Txy}(expv, new_partials)
            end,
            begin
                v = ForwardDiff.value(x)
                expv = ($f)(v, y)
                if y == zero(y) || iszero(ForwardDiff.partials(x))
                    new_partials = zero(ForwardDiff.partials(x))
                else
                    new_partials = ForwardDiff.partials(x) * y * ($f)(v, y - 1)
                end
                return ForwardDiff.Dual{Tx}(expv, new_partials)
            end,
            begin
                v = ForwardDiff.value(y)
                expv = ($f)(x, v)
                deriv = (iszero(x) && v > 0) ? zero(expv) : expv * $g(x)
                return ForwardDiff.Dual{Ty}(expv, deriv * ForwardDiff.partials(y))
            end
        )
    end
end