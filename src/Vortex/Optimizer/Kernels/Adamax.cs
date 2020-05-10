// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class Adamax : BaseOptimizer
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
#nullable enable
        public Adamax(double alpha, double b1, double b2, BaseDecay? decay = null) : base(alpha, decay)
        {
            BetaPrimary = b1;
            BetaSecondary = b2;
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
            return EOptimizerType.Adamax;
        }
    }
}
