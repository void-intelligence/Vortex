// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Nadam : Utility.BaseOptimizer
    {
        public Nadam(NadamSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.Nadam;
    }

    public sealed class NadamSettings : OptimizerSettings
    {
        public double Alpha { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Nadam;
    }
}
