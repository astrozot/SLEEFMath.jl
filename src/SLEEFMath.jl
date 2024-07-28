module SLEEFMath

import SLEEFPirates
include("sleef_derivatives.jl")

export @sleefmath

const math_fast_op =
    Dict(# basic arithmetic
         :+ => :add_fast,
         :- => :sub_fast,
         :* => :mul_fast,
         :/ => :div_fast,
         :(==) => :eq_fast,
         :!= => :ne_fast,
         :< => :lt_fast,
         :<= => :le_fast,
         :> => :gt_fast,
         :>= => :ge_fast,
         :abs => :abs_fast,
         :abs2 => :abs2_fast,
         :cmp => :cmp_fast,
         :conj => :conj_fast,
         :inv => :inv_fast,
         :rem => :rem_fast,
         :sign => :sign_fast,
         :isfinite => :isfinite_fast,
         :isinf => :isinf_fast,
         :isnan => :isnan_fast,
         :issubnormal => :issubnormal_fast,
         # math functions
         :^ => :pow_fast,
         :acos => :acos_fast,
         :acosh => :acosh_fast,
         :angle => :angle_fast,
         :asin => :asin_fast,
         :asinh => :asinh_fast,
         :atan => :atan_fast,
         :atanh => :atanh_fast,
         :cbrt => :cbrt_fast,
         :cis => :cis_fast,
         :cos => :cos_fast,
         :cosh => :cosh_fast,
         :exp10 => :exp10_fast,
         :exp2 => :exp2_fast,
         :exp => :exp_fast,
         :expm1 => :expm1_fast,
         :hypot => :hypot_fast,
         :log10 => :log10_fast,
         :log1p => :log1p_fast,
         :log2 => :log2_fast,
         :log => :log_fast,
         :max => :max_fast,
         :min => :min_fast,
         :minmax => :minmax_fast,
         :sin => :sin_fast,
         :sincos => :sincos_fast,
         :sinh => :sinh_fast,
         :sqrt => :sqrt_fast,
         :tan => :tan_fast,
         :tanh => :tanh_fast,
         # reductions
         :maximum => :maximum_fast,
         :minimum => :minimum_fast,
         :maximum! => :maximum!_fast,
         :minimum! => :minimum!_fast)

const sleef_fast_op =
    Dict(# math functions
        :^ => :pow_fast,
        :acos => :acos_fast,
        :acosh => :acosh,
        :asin => :asin_fast,
        :asinh => :asinh,
        :atan => :atan_fast,
        :atanh => :atanh,
        :cbrt => :cbrt_fast,
        :cos => :cos_fast,
        :cosh => :cosh,
        :exp10 => :exp10,
        :exp2 => :exp2,
        :exp => :exp,
        :expm1 => :expm1_fast,
        :hypot => :hypot,
        :log10 => :log10_fast,
        :log1p => :log1p,
        :log2 => :log2_fast,
        :log => :log_fast,
        :sin => :sin_fast,
        :sincos => :sincos_fast,
        :sinh => :sinh,
        :tan => :tan_fast,
        :tanh => :tanh_fast)

const rewrite_op =
    Dict(:+= => :+,
         :-= => :-,
         :*= => :*,
         :/= => :/,
         :^= => :^)

function make_sleefmath(expr::Expr)
    if expr.head === :quote
        return expr
    elseif expr.head === :call && expr.args[1] === :^
        ea = expr.args
        if length(ea) >= 3 && isa(ea[3], Int)
            # mimic Julia's literal_pow lowering of literal integer powers
            return Expr(:call, :(Base.FastMath.pow_fast), make_sleefmath(ea[2]), Val(ea[3]))
        end
    end
    op = get(rewrite_op, expr.head, :nothing)
    if op !== :nothing
        var = expr.args[1]
        rhs = expr.args[2]
        if isa(var, Symbol)
            # simple assignment
            expr = :($var = $op($var, $rhs))
        end
        # It is hard to optimize array[i += 1] += 1
        # and array[end] += 1 without bugs. (#47241)
        # We settle for not optimizing the op= call.
    end
    Base.exprarray(make_sleefmath(expr.head), Base.mapany(make_sleefmath, expr.args))
end
function make_sleefmath(symb::Symbol)
    sleef_symb = get(sleef_fast_op, symb, :nothing)
    if sleef_symb === :nothing
        fast_symb = get(math_fast_op, symb, :nothing)
        if fast_symb === :nothing
            return symb
        else
            return :(Base.FastMath.$fast_symb)
        end
    else
        return :(SLEEFMath.SLEEFPirates.$sleef_symb)
    end
end
make_sleefmath(expr) = expr

"""
    @sleefmath expr

Execute a transformed version of the expression, which calls functions that
may violate strict IEEE semantics and use the
[SLEEFPirates](https://github.com/JuliaSIMD/SLEEFPirates.jl) package. This
allows the compiler to use SIMD instructions and other optimizations that
would not be possible with strict IEEE semantics.

This macro is effective only within loops or broadcasted expressions. If used
in for loops, it often requires the use of `@simd ivdep for` loops and
`@inbounds` to be effective. For broadcasted expressions, it is often
convenient to use `@..` from the
[FastBroadcast](https://github.com/YingboMa/FastBroadcast.jl) package.

# Examples
```julia-repl
julia> @sleefmath exp(sin(3.0))
1.151562836514535

julia> xs = rand(10^6); ys = similar(xs);

julia> using BenchmarkTools, FastBroadcast

julia> @btime @.. \$ys = @sleefmath exp(sin(\$xs));
  3.708 ms (0 allocations: 0 bytes)

julia> @btime @.. \$ys = exp(sin(\$xs));
  10.265 ms (0 allocations: 0 bytes)

julia> @btime (@simd ivdep for n ∈ eachindex(\$xs); @inbounds \$ys[n] = @sleefmath exp(sin(\$xs[n])); end);
  3.719 ms (0 allocations: 0 bytes)
```
"""
macro sleefmath(expr)
    make_sleefmath(esc(expr))
end

end