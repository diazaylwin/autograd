#include <iostream>
#include <vector>

#include "rk4.h"
#include "runtime.h"
#include "autodiff.h"
#include "gradcheck.h"
#include "owned_tensor.h"
#include "optimization.h"
#include "library.h"
#include "utils.h"

// ============================================================
// Custom op example: square(x) = x^2
// ============================================================

static void square_forward(Runtime& rt, const Tensor* ins, Tensor* outs)
{
    // out = x * x
    outs[0] = mul(rt, ins[0], ins[0]);
}

static void square_backward(Runtime& rt,
                            const Tensor* primals,
                            const Tensor* seeds,
                            Tensor* grads)
{
    // primals = [x, y] where y = x^2
    // d/dx x^2 = 2x
    // grad = seed * 2 * x
    const Tensor& x = primals[0];
    const Tensor& seed = seeds[0];

    Tensor two = from_vector(rt, {1}, {2.0});
    Tensor two_x = mul(rt, expand(rt, two, x), x);
    grads[0] = mul(rt, seed, two_x);
}

static void test_custom()
{
    std::cout << "=== Custom op test ===\n";

    // Define custom op
    CustomOp square_op;
    square_op.forward = square_forward;
    square_op.backward = square_backward;
    square_op.num_inputs = 1;
    square_op.num_outputs = 1;

    // Build: y = sum(square(x))
    Builder b;
    Var x = Input(b);
    Var sq = Custom(b, square_op, {x});
    Var y = Sum(b, sq);
    Program fwd = Finalize(b, {y});

    CompiledProgram cp;
    cp.fwd = fwd;
    cp.bwd = build_vjp(cp.fwd);
    Optimise(cp);

    Runtime rt;
    runtime_init(&rt, (uint64_t)1 << 16);

    // Input: small vector
    OwnedTensor x_in = owned_from_vector({4}, {1.0, 2.0, 3.0, 4.0});

    std::vector<OwnedTensor> inputs;
    inputs.push_back(x_in);

    // Seed for scalar output
    OwnedTensor seed = owned_from_vector({1}, {1.0});
    std::vector<OwnedTensor> seeds;
    seeds.push_back(seed);

    // Forward
    std::cout << "  executing forward...\n";
    runtime_reset(&rt);
    std::vector<Tensor> rt_inputs;
    rt_inputs.push_back(from_vector(rt, x_in.sizes, x_in.data));

    Tape tape;
    std::vector<Tensor> outs = execute(cp.fwd, rt, tape, rt_inputs);
    std::cout << "  forward done\n";
    OwnedTensor yT = from_tensor(rt, outs[0]);
    std::cout << "sum(square(x)) = " << yT.data[0] << "\n";  // 1+4+9+16 = 30

    // Gradcheck
    GradcheckOptions opt;
    opt.eps  = 1e-6;
    opt.rtol = 1e-5;
    opt.atol = 1e-8;
    opt.seed = 42u;
    opt.mode = GRADCHECK_COORDINATE;
    opt.u.coord.max_coords_per_input = 0;

    GradcheckReport rep = gradcheck(cp, rt, inputs, seeds, opt);

    if (rep.ok)
        std::cout << "custom op gradcheck: OK\n";
    else
    {
        std::cout << "custom op gradcheck: FAIL\n";
        std::cout << "  input_index = " << rep.input_index << "\n";
        std::cout << "  index       = " << rep.index << "\n";
        std::cout << "  ad          = " << rep.ad << "\n";
        std::cout << "  fd          = " << rep.fd << "\n";
        std::cout << "  abs_err     = " << rep.abs_err << "\n";
    }

    runtime_free(&rt);
    std::cout << "\n";
}

static void test_sigmoid()
{
    std::cout << "=== Sigmoid test ===\n";

    // Build: y = sum(sigmoid(x))
    Builder b;
    Var x = Input(b);
    Var s = Sigmoid(b, x);
    Var y = Sum(b, s);
    Program fwd = Finalize(b, {y});

    CompiledProgram cp;
    cp.fwd = fwd;
    cp.bwd = build_vjp(cp.fwd);
    Optimise(cp);

    Runtime rt;
    runtime_init(&rt, (uint64_t)1 << 16);

    // Input: small vector
    OwnedTensor x_in = owned_from_vector({4}, {-1.0, 0.0, 1.0, 2.0});

    std::vector<OwnedTensor> inputs;
    inputs.push_back(x_in);

    // Seed for scalar output
    OwnedTensor seed = owned_from_vector({1}, {1.0});
    std::vector<OwnedTensor> seeds;
    seeds.push_back(seed);

    // Forward
    runtime_reset(&rt);
    std::vector<Tensor> rt_inputs;
    rt_inputs.push_back(from_vector(rt, x_in.sizes, x_in.data));

    Tape tape;
    std::vector<Tensor> outs = execute(cp.fwd, rt, tape, rt_inputs);
    OwnedTensor yT = from_tensor(rt, outs[0]);
    std::cout << "sum(sigmoid(x)) = " << yT.data[0] << "\n";

    // Gradcheck
    GradcheckOptions opt;
    opt.eps  = 1e-6;
    opt.rtol = 1e-5;
    opt.atol = 1e-8;
    opt.seed = 42u;
    opt.mode = GRADCHECK_COORDINATE;
    opt.u.coord.max_coords_per_input = 0;

    GradcheckReport rep = gradcheck(cp, rt, inputs, seeds, opt);

    if (rep.ok)
        std::cout << "sigmoid gradcheck: OK\n";
    else
    {
        std::cout << "sigmoid gradcheck: FAIL\n";
        std::cout << "  input_index = " << rep.input_index << "\n";
        std::cout << "  index       = " << rep.index << "\n";
        std::cout << "  ad          = " << rep.ad << "\n";
        std::cout << "  fd          = " << rep.fd << "\n";
        std::cout << "  abs_err     = " << rep.abs_err << "\n";
    }

    runtime_free(&rt);
    std::cout << "\n";
}

int main()
{
    // Test custom op
    test_custom();

    // Test library function
    test_sigmoid();

    // 1) Build the IR program
    Program prog = build_rk4_program(100);

    // 2) Compile: build VJP + optimise
    CompiledProgram cp;
    cp.fwd = prog;
    cp.bwd = build_vjp(cp.fwd);
    Optimise(cp);

    // 3) Create runtime
    Runtime rt;
    runtime_init(&rt, (uint64_t)1 << 20);
    runtime_reset(&rt);

    // 3) Inputs (scalars = rank-1 {1})
    const double y0    = 1.0;
    const double t0    = 0.0;
    const double dt    = 0.01;
    const double theta = -1.0;

    OwnedTensor y0_in = owned_from_vector({1}, {y0});
    OwnedTensor t0_in = owned_from_vector({1}, {t0});
    OwnedTensor dt_in = owned_from_vector({1}, {dt});
    OwnedTensor theta_in = owned_from_vector({1}, {theta});

    std::vector<OwnedTensor> inputs;
    inputs.push_back(y0_in);
    inputs.push_back(t0_in);
    inputs.push_back(dt_in);
    inputs.push_back(theta_in);

    // 4) Forward once (optional, just to print yT)
    runtime_reset(&rt);

    std::vector<Tensor> rt_inputs;
    rt_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        rt_inputs.push_back(from_vector(rt, inputs[i].sizes, inputs[i].data));

    Tape tape;
    std::vector<Tensor> outs = execute(cp.fwd, rt, tape, rt_inputs);
    require(outs.size() == 1);

    OwnedTensor yT = from_tensor(rt, outs[0]);
    std::cout << "yT = " << yT.data[0] << "\n";

    // 5) Seeds for objective: F = dot(out, seed). For scalar output, seed = {1}.
    OwnedTensor seed;
    seed.offset  = 0;
    seed.sizes   = std::vector<uint64_t>{1};
    seed.strides = std::vector<uint64_t>{1};
    seed.data    = std::vector<double>{1.0};

    std::vector<OwnedTensor> seeds;
    seeds.push_back(seed);

    // 6) Gradcheck
    GradcheckOptions opt;
    opt.eps  = 1e-6;
    opt.rtol = 1e-5;
    opt.atol = 1e-8;

    opt.seed = 123u;
    opt.mode = GRADCHECK_COORDINATE;
    opt.u.coord.max_coords_per_input = 0; // all coords

    GradcheckReport rep = gradcheck(cp, rt, inputs, seeds, opt);

    if (rep.ok)
    {
        std::cout << "gradcheck: OK\n";
        std::cout << "input_index = " << rep.input_index << "\n";
        std::cout << "index       = " << rep.index << "\n";
        std::cout << "ad          = " << rep.ad << "\n";
        std::cout << "fd          = " << rep.fd << "\n";
        std::cout << "abs_err     = " << rep.abs_err << "\n";
        std::cout << "rel_err     = " << rep.rel_err << "\n";
    }
    else
    {
        std::cout << "gradcheck: FAIL\n";
        std::cout << "input_index = " << rep.input_index << "\n";
        std::cout << "index       = " << rep.index << "\n";
        std::cout << "ad          = " << rep.ad << "\n";
        std::cout << "fd          = " << rep.fd << "\n";
        std::cout << "abs_err     = " << rep.abs_err << "\n";
        std::cout << "rel_err     = " << rep.rel_err << "\n";
    }

    runtime_free(&rt);
    return 0;
}
