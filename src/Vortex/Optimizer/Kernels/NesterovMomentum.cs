// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class NesterovMomentum : BaseOptimizer
    {       
        /// <summary>
        /// Momentum Parameter
        /// </summary>
        public double Tao { get; set; }

#nullable enable
        public NesterovMomentum(double alpha = 0.001, BaseDecay? decay = null, double tao = 0.9) : base(alpha, decay)
        {
            Tao = tao;
        }
#nullable disable

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
            {
                // Cache 1
                dJdX.Cache.Add(dJdX.Fill(0));

                // Old Delta
                dJdX.Cache.Add(dJdX.Fill(0));

                // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));
            }

            // Iteration T
            dJdX.Cache[^1][0, 0]++;

            dJdX.Cache[1] = dJdX.Cache[0];
            dJdX.Cache[0] = -Alpha * dJdX + Tao * dJdX.Cache[0];
            return x - (1.0 + Tao) * dJdX.Cache[0] - Tao * dJdX.Cache[1];
        }


        public override EOptimizerType Type()
        {
            return EOptimizerType.NesterovMomentum;
        }
    }
}


