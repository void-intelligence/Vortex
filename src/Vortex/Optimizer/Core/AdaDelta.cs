// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class AdaDelta : Utility.BaseOptimizer
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }

        public AdaDelta(AdaDeltaSettings settings) : base(settings)
        {
            Rho = settings.Rho;
            Epsilon = settings.Epsilon;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.AdaDelta;
    }

    public sealed class AdaDeltaSettings : OptimizerSettings
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.AdaDelta;

        public AdaDeltaSettings(double alpha, double rho, double epsilon) : base(alpha)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
    }
}
