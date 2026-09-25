using SLEEFMath, SLEEFPirates
using Test
using Chairmarks
using ForwardDiff
import Zygote
@testset "SLEEFMath.jl" verbose=true begin
    @testset "Consistency" begin
        @test (@sleefmath acos(0.31^2.5)) ≈ acos(0.31^2.5)
        @test (@sleefmath asinh(cbrt(7.3))) ≈ asinh(cbrt(7.3))
        @test (@sleefmath atan(atanh(asinh(0.5)))) ≈ atan(atanh(asinh(0.5)))
        @test (@sleefmath exp(cos(3.0))) ≈ exp(cos(3.0))
        @test (@sleefmath exp10(cosh(3.0))) ≈ exp10(cosh(3.0))
        @test (@sleefmath exp2(sinh(3.0))) ≈ exp2(sinh(3.0))
        @test (@sleefmath expm1(tan(3.0))) ≈ expm1(tan(3.0))
        for x in (-745.0, 710.0, -100.0f0, 100.0f0)
            @test (@sleefmath expm1(x)) === expm1(x)
        end
        for x in (NaN, Inf, -Inf, 1.0, 1e-310, NaN32, Inf32, 1.0f0)
            @test (@sleefmath isnan(x)) === isnan(x)
            @test (@sleefmath isinf(x)) === isinf(x)
            @test (@sleefmath isfinite(x)) === isfinite(x)
            @test (@sleefmath issubnormal(x)) === issubnormal(x)
        end
        # runtime (non-literal) exponents
        for x in (-2.0, -2.0f0, 1.5, -1.5f0), p in (3, -2, 2.0, -3.0f0, 2.0^60)
            @test (@sleefmath x^p) ≈ x^p
            @test typeof(@sleefmath x^p) === typeof(x^p)
        end
        for (x, y) in ((0.0, 0.0), (NaN, 0.0), (0.0, -1.0),
                       (-0.0, -3.0), (-0.0, -2.0), (-0.0, 3.0), (-Inf, 3.0), (-Inf, 2.0),
                       (0.5, Inf), (2.0, -Inf), (NaN32, 0.0f0), (-0.0f0, -3.0f0))
            @test isequal((@sleefmath x^y), x^y)
        end
        let x = -2.0, y = 0.5
            @test isnan(@sleefmath x^y)
        end
        let x = -Inf, y = -3.0
            # IEEE/C give -0.0; Base's Float64 method returns +0.0
            @test (@sleefmath x^y) == 0
        end
        let x = 2, n = 10
            @test (@sleefmath x^n) === 1024
        end
        @test (@sleefmath log10(log(hypot(3.0, 4.0)))) ≈ log10(log(hypot(3.0, 4.0)))
        @test (@sleefmath log1p(log2(3.0))) ≈ log1p(log2(3.0))
        @test (@sleefmath max(min(3.0, 4.0), 5.0)) ≈ max(min(3.0, 4.0), 5.0)
        @test all((@sleefmath sincos(sin(3.0))) .≈ sincos(sin(3.0)))
        @test (@sleefmath sinh(sqrt(3.0))) ≈ sinh(sqrt(3.0))
        x = y = 1.23
        a = (@sleefmath x += sin(max(3.0, 4.0)))
        b = (y += sin(max(3.0, 4.0)))
        @test a ≈ b
        @test (@sleefmath sum([1,2,3])) == 6
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

    @testset "ForwardDiff" begin
        D = ForwardDiff.derivative
        @test all(SLEEFMath.DiffRules.hasdiffrule(m, f, n) for (m, f, n) in
                  ((:SLEEFPirates, :tan, 1), (:SLEEFPirates, :hypot, 2), (:SLEEFMath, :pow_fast, 2)))
        let (∂x, ∂y) = SLEEFMath.DiffRules.diffrule(:SLEEFMath, :pow_fast, :x, :y)
            @test (@eval let x = -2.0, y = 3.0; $∂x end) ≈ 12.0
            @test (@eval let x = 2.0, y = 3.0; $∂y end) ≈ 8log(2.0)
        end
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
        n, y = 3, 3.0
        @test D(x -> (@sleefmath x^n), -2.0) ≈ 12.0
        @test D(x -> (@sleefmath x^y), -2.0) ≈ 12.0
        @test D(y -> (@sleefmath 2.0^y), 1.5) ≈ 2.0^1.5 * log(2.0)
        # non-fast versions
        @test D(x -> (SLEEFPirates.sin(SLEEFPirates.hypot(x, 2.5))), 4.1) ≈ D(x -> sin(hypot(x, 2.5)), 4.1)
        @test D(x -> (SLEEFPirates.acos(SLEEFPirates.pow(x, 2.5))), 0.31) ≈ D(x -> acos(x^2.5), 0.31)
        @test D(x -> (SLEEFPirates.atan(SLEEFPirates.cbrt(x))), 7.3) ≈ D(x -> atan(cbrt(x)), 7.3)
        @test D(x -> (SLEEFPirates.expm1(SLEEFPirates.tan(x))), 3.0) ≈ D(x -> expm1(tan(x)), 3.0)
        @test D(x -> (SLEEFPirates.log10(SLEEFPirates.log(SLEEFPirates.cos(x) + 10))), 3.0) ≈ D(x -> log10(log(cos(x) + 10)), 3.0)
        @test D(x -> (SLEEFPirates.sincos(SLEEFPirates.log2(x)))[1], 32.1) ≈ D(x -> sincos(log2(x))[1], 32.1)
        @test D(x -> (SLEEFPirates.sincos(SLEEFPirates.log2(x)))[2], 32.1) ≈ D(x -> sincos(log2(x))[2], 32.1)
    end

    @testset "Zygote" begin
        D = (f, x) -> f'(x)
        @test all(Zygote.gradient((x, y) -> (@sleefmath hypot(x, y)), 3.0, 4.0) .≈ (0.6, 0.8))
        @test Zygote.gradient((x, y) -> (@sleefmath hypot(x, y)), 0.0, 0.0) == (0.0, 0.0)
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
        n, y = 3, 3.0
        @test D(x -> (@sleefmath x^n), -2.0) ≈ 12.0
        @test D(x -> (@sleefmath x^y), -2.0) ≈ 12.0
        @test D(y -> (@sleefmath 2.0^y), 1.5) ≈ 2.0^1.5 * log(2.0)
        # non-fast versions
        @test_broken D(x -> (SLEEFPirates.acos(SLEEFPirates.pow(x, 2.5))), 0.31) ≈ D(x -> acos(x^2.5), 0.31)
        @test D(x -> (SLEEFPirates.atan(SLEEFPirates.cbrt(x))), 7.3) ≈ D(x -> atan(cbrt(x)), 7.3)
        @test D(x -> (SLEEFPirates.expm1(SLEEFPirates.tan(x))), 3.0) ≈ D(x -> expm1(tan(x)), 3.0)
        @test D(x -> (SLEEFPirates.log10(SLEEFPirates.log(SLEEFPirates.cos(x) + 10))), 3.0) ≈ D(x -> log10(log(cos(x) + 10)), 3.0)
        @test D(x -> (SLEEFPirates.sincos(SLEEFPirates.log2(x)))[1], 32.1) ≈ D(x -> sincos(log2(x))[1], 32.1)
        @test D(x -> (SLEEFPirates.sincos(SLEEFPirates.log2(x)))[2], 32.1) ≈ D(x -> sincos(log2(x))[2], 32.1)
    end

end;
