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

        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            return (Alpha * (w.Hadamard(dJdW)));
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            return (Alpha * (b.Hadamard(dJdB)));
        }

        public override EOptimizerType Type() => EOptimizerType.GradientDescent;
    }

    public sealed class GradientDescent : Utility.Optimizer
    {
        public override EOptimizerType Type() => EOptimizerType.GradientDescent;

        public GradientDescent(double alpha = 0.001) : base(alpha)
        {
        }
    }
}
