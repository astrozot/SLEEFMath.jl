const SLEEFFloat = Union{Float32,Float64}

"""
    SLEEFMath.pow_fast(x, y)

Fast `x^y` used by `@sleefmath` for non-literal exponents.

It is built on `SLEEFPirates.pow_fast` (i.e. `exp2(y * log2(x))`, with the same
accuracy), but, unlike it, handles negative bases with integer-valued exponents
(`(-2.0)^3 == -8.0`), `x^0 == 1` (including `0^0` and `NaN^0`), and the sign of
`(-0.0)^y` and `(-Inf)^y`, using branch-free code so that loops still vectorize.
For speed, `1^NaN` and `(±1)^±Inf` are not special-cased and return `NaN`
(Base returns `1`). Arguments that are not `Float32`/`Float64` fall back to
`Base.:^`.
"""
@inline function pow_fast(x::T, y::T) where {T<:SLEEFFloat}
    h = y * T(0.5)
    yisint = trunc(y) == y
    yisodd = yisint & (trunc(h) != h)
    # SLEEFPirates.pow_fast(|x|, y) already handles |x| ∈ {0, Inf}; fix the sign
    r = flipsign(SLEEFPirates.pow_fast(abs(x), y), ifelse(yisodd, x, one(T)))
    r = ifelse((x < 0) & !yisint, T(NaN), r)
    return ifelse(y == 0, one(T), r)
end
@inline pow_fast(x::SLEEFFloat, n::Integer) = pow_fast(x, oftype(x, n))
@inline pow_fast(x::Union{Integer,SLEEFFloat}, y::SLEEFFloat) = pow_fast(promote(x, y)...)
@inline pow_fast(x, y) = x^y
