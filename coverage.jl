using Coverage
using Pkg

clean_folder("src")
clean_folder("test")

Pkg.activate(".")
try
  Pkg.test("SLEEFMath"; coverage=true, test_args=ARGS, julia_args=["-t", "auto"])
catch
end
Pkg.activate()

coverage = process_folder("src");
@info "Writing coverage file `lcov.info`..."
open("lcov.info", "w") do io
  LCOV.write(io, coverage)
end;

clean_folder("src")
clean_folder("test")
