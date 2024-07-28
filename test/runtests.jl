using SLEEFMath
using Test
using Chairmarks
using ForwardDiff

@testset "SLEEFMath.jl" verbose=true begin
    @testset "Consistency" begin
        @test (@sleefmath acos(0.31^2.5)) ≈ acos(0.31^2.5)
        @test (@sleefmath asinh(cbrt(7.3))) ≈ asinh(cbrt(7.3))
        @test (@sleefmath atan(atanh(asinh(0.5)))) ≈ atan(atanh(asinh(0.5)))
        @test (@sleefmath exp(cos(3.0))) ≈ exp(cos(3.0))
        @test (@sleefmath exp10(cosh(3.0))) ≈ exp10(cosh(3.0))
        @test (@sleefmath exp2(sinh(3.0))) ≈ exp2(sinh(3.0))
        @test (@sleefmath expm1(tan(3.0))) ≈ expm1(tan(3.0))
        @test (@sleefmath log10(log(hypot(3.0, 4.0)))) ≈ log10(log(hypot(3.0, 4.0)))
        @test (@sleefmath log1p(log2(3.0))) ≈ log1p(log2(3.0))
        @test (@sleefmath max(min(3.0, 4.0), 5.0)) ≈ max(min(3.0, 4.0), 5.0)
        @test all((@sleefmath sincos(sin(3.0))) .≈ sincos(sin(3.0)))
        @test (@sleefmath sinh(sqrt(3.0))) ≈ sinh(sqrt(3.0))
    end

    @testset "Performance" begin
        xs = rand(10^6)
        ys = similar(xs)
        t1 = @b (@simd ivdep for n ∈ eachindex($xs)
            @inbounds $ys[n] = @sleefmath exp(sin($xs[n]))
        end)
        t2 = @b (@simd ivdep for n ∈ eachindex($xs)
            @inbounds $ys[n] = exp(sin($xs[n]))
        end)
        @info "SLEEFMath.jl vs. Base: $(t1.time) vs. $(t2.time)"
        @test_skip t1.time < t2.time
        t1 = @b (@simd ivdep for n ∈ eachindex($xs)
            @inbounds $ys[n] = @sleefmath expm1(atan($xs[n]))^1.23
        end)
        t2 = @b (@simd ivdep for n ∈ eachindex($xs)
            @inbounds $ys[n] = expm1(atan($xs[n]))^1.23
        end)
        @info "SLEEFMath.jl vs. Base: $(t1.time) vs. $(t2.time)"
        @test_skip t1.time < t2.time
        t1 = @b (@simd ivdep for n ∈ eachindex($xs)
            @inbounds $ys[n] = @sleefmath log1p($xs[n])
        end)
        t2 = @b (@simd ivdep for n ∈ eachindex($xs)
            @inbounds $ys[n] = log1p($xs[n])
        end)
        @info "SLEEFMath.jl vs. Base: $(t1.time) vs. $(t2.time)"
        @test_skip t1.time < t2.time
    end

    @testset "Derivatives" begin
        D = ForwardDiff.derivative
        @test D(x -> (@sleefmath acos(x^2.5)), 0.31) ≈ D(x -> acos(x^2.5), 0.31)
        @test D(x -> (@sleefmath asinh(cbrt(x))), 7.3) ≈ D(x -> asinh(cbrt(x)), 7.3)
        @test D(x -> (@sleefmath atan(atanh(asinh(x)))), 0.5) ≈ D(x -> atan(atanh(asinh(x))), 0.5)
        @test D(x -> (@sleefmath exp(cos(x))), 3.0) ≈ D(x -> exp(cos(x)), 3.0)
        @test D(x -> (@sleefmath exp10(cosh(x))), 3.0) ≈ D(x -> exp10(cosh(x)), 3.0)
        @test D(x -> (@sleefmath exp2(sinh(x))), 3.0) ≈ D(x -> exp2(sinh(x)), 3.0)
        @test D(x -> (@sleefmath expm1(tan(x))), 3.0) ≈ D(x -> expm1(tan(x)), 3.0)
        @test D(x -> (@sleefmath log10(log(hypot(x, 4.0)))), 3.0) ≈ D(x -> log10(log(hypot(x, 4.0))), 3.0)
        @test D(x -> (@sleefmath log10(log(hypot(3.0, x)))), 4.0) ≈ D(x -> log10(log(hypot(3.0, x))), 4.0)
        @test D(x -> (@sleefmath log10(log(hypot(x, x^2)))), 2.0) ≈ D(x -> log10(log(hypot(x, x^2))), 2.0)
        @test D(x -> (@sleefmath log1p(log2(x))), 7.9) ≈ D(x -> log1p(log2(x)), 7.9)
        @test D(x -> (@sleefmath sincos(sin(x)))[1], 32.1) ≈ D(x -> sincos(sin(x))[1], 32.1)
        @test D(x -> (@sleefmath sincos(cos(x)))[2], 32.1) .≈ D(x -> sincos(cos(x))[2], 32.1)
        @test D(x -> (@sleefmath sinh(sqrt(x))), 3.0) ≈ D(x -> sinh(sqrt(x)), 3.0)
    end
end;
