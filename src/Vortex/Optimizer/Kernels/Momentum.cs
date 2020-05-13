// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class Momentum : BaseOptimizer
    {
        /// <summary>
        /// Momentum Parameter
        /// </summary>
        public double Tao { get; set; }

#nullable enable
        public Momentum(double alpha = 0.01, IDecay? decay = null, double tao = 0.9) : base(alpha, decay)
        {
            Tao = tao;
        }
#nullable disable

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
            {
                dJdX.Cache.Add(dJdX.Fill(0));

                // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));
            }

            // Iteration T
            dJdX.Cache[^1][0, 0]++;

            dJdX.Cache[0] = Tao * dJdX.Cache[0] + (1 - Tao) * dJdX;
            return x - Alpha * dJdX.Cache[0];
        }

        public override EOptimizerType Type()
        {
            return EOptimizerType.Momentum;
        }
    }
}