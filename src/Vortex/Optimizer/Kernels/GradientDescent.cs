// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class GradientDescentKernel : BaseOptimizerKernel
    {
        public GradientDescentKernel(GradientDescent settings) : base(settings)
        {
        }

        public GradientDescentKernel(double alpha = 0.001) : base(new GradientDescent(alpha))
        {
        }

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
            {
                // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));
            }

            // Iteration T
            dJdX.Cache[^1][0, 0]++;

            return (Alpha * (x.Hadamard(dJdX)));
        }

        public override EOptimizerType Type() => EOptimizerType.GradientDescent;
    }

    public sealed class GradientDescent : Utility.BaseOptimizer
    {
        public override EOptimizerType Type() => EOptimizerType.GradientDescent;

        public GradientDescent(double alpha = 0.001) : base(alpha)
        {
        }
    }
}
