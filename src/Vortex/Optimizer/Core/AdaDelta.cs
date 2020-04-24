// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class AdaDelta : Utility.BaseOptimizer
    {
        public AdaDelta(AdaDeltaSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.AdaDelta;
    }

    public sealed class AdaDeltaSettings : OptimizerSettings
    {
        public double Alpha { get; set; }
        public double Rho { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.AdaDelta;
    }
}
