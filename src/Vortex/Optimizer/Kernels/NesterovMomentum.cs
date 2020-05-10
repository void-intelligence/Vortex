// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class NesterovMomentum : BaseOptimizer
    {
#nullable enable
        public NesterovMomentum(double alpha = 0.001, BaseDecay? decay = null) : base(alpha, decay)
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
            return null;
        }


        public override EOptimizerType Type()
        {
            return EOptimizerType.NesterovMomentum;
        }
    }
}


