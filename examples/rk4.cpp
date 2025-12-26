#include "rk4.h"
#include "symbols.h"

// rhs(t,y,theta) but t unused here
static Var rhs_linear(const Var& y, const Var& theta)
{
    return theta * y;
}

// one RK4 step, purely symbolic
static Var rk4_step(const Var& y, const Var& dt, const Var& theta)
{
    const Var half = Constant(*y.b, 0.5);
    const Var two  = Constant(*y.b, 2.0);
    const Var six  = Constant(*y.b, 6.0);

    const Var k1 = rhs_linear(y, theta);
    const Var k2 = rhs_linear(y + (half * dt) * k1, theta);
    const Var k3 = rhs_linear(y + (half * dt) * k2, theta);
    const Var k4 = rhs_linear(y + dt * k3, theta);

    const Var ynext = y + (dt / 6.0) * (k1 + two * k2 + two * k3 + k4);
    return ynext;
}

// Build the Scan body program: inputs = [state y, invariants dt, theta], outputs = [y_next]
static Program build_rk4_body()
{
    Builder bb;

    Var y     = Input(bb);  // carried state
    Var dt    = Input(bb);  // invariant #0
    Var theta = Input(bb);  // invariant #1

    Var y_next = rk4_step(y, dt, theta);
    return Finalize(bb, std::vector<Var>{ y_next });
}

Program build_rk4_program(const uint64_t nsteps)
{
    Builder b;

    // keep your original signature (t0 is still an input for compatibility)
    const Var y0    = Input(b);
    const Var t0    = Input(b);   (void)t0; // unused in this RHS
    const Var dt    = Input(b);
    const Var theta = Input(b);

    const Program body = build_rk4_body();

    // Scan repeats the body nsteps times carrying y forward
    const Var yT = Scan(b, y0, body, (uint32_t)nsteps, std::vector<Var>{ dt, theta });

    return Finalize(b, std::vector<Var>{ yT });
}