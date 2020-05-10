// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System.Runtime.CompilerServices;
using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class MomentumKernel : BaseOptimizerKernel
    {
        /// <summary>
        /// Momentum Parameter
        /// </summary>
        public double Tao { get; set; }

        public MomentumKernel(Momentum settings) : base(settings)
        {
            Tao = settings.Tao;
        }

        public MomentumKernel(double alpha = 0.01, double tao = 0.9) : base(new Momentum(alpha, tao))
        {
        }

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

            dJdX.Cache[0] = Tao * dJdX.Cache[0] + ((1 - Tao) * dJdX);
            return (x - Alpha * dJdX.Cache[0]);
        }

        public override EOptimizerType Type() => EOptimizerType.Momentum;
    }

    public sealed class Momentum : Utility.BaseOptimizer
    {
        public double Tao { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Momentum;

        public Momentum(double alpha = 0.01, double tao = 0.9) : base(alpha)
        {
            Tao = tao;
        }
    }
}