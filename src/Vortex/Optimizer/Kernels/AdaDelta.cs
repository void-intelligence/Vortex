// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class AdaDeltaKernel : BaseOptimizerKernel
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }

        public AdaDeltaKernel(AdaDelta settings) : base(settings)
        {
            Rho = settings.Rho;
            Epsilon = settings.Epsilon;
        }
        
        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.AdaDelta;
    }

    public sealed class AdaDelta : BaseOptimizer
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.AdaDelta;

        public AdaDelta(double rho, double epsilon, double alpha = 0.001) : base(alpha)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
    }
}
