// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
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
        public NesterovMomentum(double alpha = 0.001, IDecay? decay = null, double tao = 0.9) : base(alpha, decay)
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
            dJdX.Cache[0] = dJdX * -Alpha + dJdX.Cache[0] * Tao;
            return x - dJdX.Cache[0] * (1.0 + Tao) - dJdX.Cache[1] * Tao;
        }

        public override EOptimizerType Type()
        {
            return EOptimizerType.NesterovMomentum;
        }
    }
}


