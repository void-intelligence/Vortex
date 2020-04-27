// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Nadam : Utility.BaseOptimizer
    {
        public Nadam(NadamSettings settings) : base(settings)
        {
        }
        public Nadam(double alpha = 0.001) : base(new NadamSettings(alpha))
        {
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.Nadam;
    }

    public sealed class NadamSettings : OptimizerSettings
    {
        public override EOptimizerType Type() => EOptimizerType.Nadam;

        public NadamSettings(double alpha = 0.001) : base(alpha)
        {
        }
    }
}
