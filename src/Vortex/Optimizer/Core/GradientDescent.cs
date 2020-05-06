// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class GradientDescent : Utility.BaseOptimizer
    {
        public GradientDescent(GradientDescentSettings settings) : base(settings)
        {
        }

        public GradientDescent(double alpha = 0.001) : base(new GradientDescentSettings(alpha))
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

    public sealed class GradientDescentSettings : OptimizerSettings
    {
        public override EOptimizerType Type() => EOptimizerType.GradientDescent;

        public GradientDescentSettings(double alpha = 0.001) : base(alpha)
        {
        }
    }
}
