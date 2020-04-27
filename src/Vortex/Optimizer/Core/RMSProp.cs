// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class RMSProp : Utility.BaseOptimizer
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }

        public RMSProp(RMSPropSettings settings) : base(settings)
        {
            Rho = settings.Rho;
            Epsilon = settings.Epsilon;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.RMSProp;
    }

    public sealed class RMSPropSettings : OptimizerSettings
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.RMSProp;

        public RMSPropSettings(double rho, double epsilon, double alpha = 0.001) : base(alpha)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
    }
}
