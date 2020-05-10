// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class AdaDelta : BaseOptimizer
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }

#nullable enable
        public AdaDelta(double alpha, double rho, BaseDecay? decay = null, double epsilon = 0.00001) : base(alpha, decay)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
#nullable disable

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
            // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));

            // Iteration T
            dJdX.Cache[^1][0, 0]++;
            return null;
        }

        public override EOptimizerType Type()
        {
            return EOptimizerType.AdaDelta;
        }
    }
}
