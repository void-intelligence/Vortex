// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Adam : Utility.BaseOptimizer
    {
        public Adam(AdamSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.Adam;
    }

    public sealed class AdamSettings : OptimizerSettings
    {
        public double Alpha { get; set; }
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public double BetaPrimary_T { get; set; }
        public double BetaSecondary_T { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Adam;
    }
}
