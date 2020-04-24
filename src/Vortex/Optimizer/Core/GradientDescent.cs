// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class GradientDescent : Utility.BaseOptimizer
    {
        public GradientDescent(GradientDescentSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }
        public override string ToString() => Type().ToString();

        public override EOptimizerType Type() => EOptimizerType.GradientDescent;

        public override Matrix CalculateDeltaW(Matrix W, Matrix dJdW)
        {
            return (Alpha * (W.Transpose().Hadamard(dJdW)));
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdb)
        {
            return (Alpha * (b.Hadamard(dJdb)));
        }
    }

    public sealed class GradientDescentSettings : OptimizerSettings
    {
        public double Alpha { get; private set; }

        public GradientDescentSettings(double alpha)
        {
            Alpha = alpha;
            public override EOptimizerType Type() => EOptimizerType.GradientDescent;
    }
    }
}
