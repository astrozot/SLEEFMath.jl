# SLEEFMath

[![Build Status](https://github.com/astrozot/SLEEFMath.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/astrozot/SLEEFMath.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/astrozot/SLEEFMath.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/astrozot/SLEEFMath.jl)

This package provides two main functionalities:

- a macro `@sleefmath` that works similarly to `@fastmath`: it transform
  an mathematical expression into a faster version based on the 
  [SLEEFPirates](https://github.com/JuliaSIMD/SLEEFPirates.jl);

- a set of differentiation rules based on
  [ChainRules](https://github.com/JuliaDiff/ChainRules.jl) and also
  working with [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl).

## How it works

A call to `@sleefmath expr` generates a transformed version of the expression,
which calls functions that may violate strict IEEE semantics and use the
[SLEEFPirates](https://github.com/JuliaSIMD/SLEEFPirates.jl) package. This
allows the compiler to use SIMD instructions and other optimizations that
would not be possible with strict IEEE semantics.

This macro is effective only within loops or broadcasted expressions. If used
in for loops, it often requires the use of `@simd ivdep for` loops and
`@inbounds` to be effective. For broadcasted expressions, it is often
convenient to use `@..` from the
[FastBroadcast](https://github.com/YingboMa/FastBroadcast.jl) package.

### Examples
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

## Limitation

The current version of this package implements real-valued mathematical
functions based on the SLEEF library. Future versions will also implement
functions with complex arguments.
