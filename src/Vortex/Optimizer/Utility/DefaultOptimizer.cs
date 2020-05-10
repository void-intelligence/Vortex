// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Decay.Kernels;
using Vortex.Decay.Utility;

namespace Vortex.Optimizer.Utility
{
    public sealed class DefaultOptimizer : BaseOptimizer
    {
#nullable enable
        public DefaultOptimizer(double alpha = 0.01, BaseDecay? decay = null) : base(alpha, decay)
        {
        }
#nullable disable

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
                // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));

            // Iteration T
            dJdX.Cache[^1][0, 0]++;

            return Alpha * x.Hadamard(dJdX);
        }

        public override EOptimizerType Type()
        {
            return EOptimizerType.Default;
        }
    }
}