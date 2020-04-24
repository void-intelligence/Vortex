// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Adamax : Utility.BaseOptimizer
    {
        public Adamax(AdamaxSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.Adamax;
    }

    public sealed class AdamaxSettings : OptimizerSettings
    {
        public double Alpha { get; set; }
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Adamax;
    }
}
