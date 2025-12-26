#include "rk4.h"
#include "symbols.h"

static Var rhs_linear(const Var& /*t*/, const Var& y, const Var& theta)
{
    return theta * y;
}

static Var rk4_step(
    const Var& t,
    const Var& y,
    const Var& dt,
    const Var& theta
)
{
    const Var k1 = rhs_linear(t, y, theta);
    const Var k2 = rhs_linear(t + 0.5 * dt, y + (0.5 * dt) * k1, theta);
    const Var k3 = rhs_linear(t + 0.5 * dt, y + (0.5 * dt) * k2, theta);
    const Var k4 = rhs_linear(t + dt, y + dt * k3, theta);
    const Var ynext = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    return ynext;
}

static Var rk4_IVP(
    const Var& y0,
    const Var& t0,
    const Var& dt,
    const Var& theta,
    const uint64_t nsteps
)
{    
    Var y = y0;
    Var t = t0;

    for (uint64_t i = 0; i < nsteps; ++i)
    {
        y = rk4_step(t, y, dt, theta);
        t = t + dt;
    }
    return y;
}

Program build_rk4_program(const uint64_t nsteps)
{
    Builder b;

    const Var y0    = Input(b);
    const Var t0    = Input(b);
    const Var dt    = Input(b);
    const Var theta = Input(b);

    const Var yT = rk4_IVP(y0, t0, dt, theta, nsteps);
    return Finalize(b, std::vector<Var>{ yT });
}
