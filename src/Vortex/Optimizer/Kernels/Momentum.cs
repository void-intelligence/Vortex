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

        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            if (dJdW.Cache.Count == 0)
            {
                dJdW.Cache.Add(dJdW.Fill(0));
            }

            dJdW.Cache[0] = Tao * dJdW.Cache[0] + ((1 - Tao) * dJdW);
            return (w - Alpha * dJdW.Cache[0]);
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            if (dJdB.Cache.Count == 0)
            {
                dJdB.Cache.Add(dJdB.Fill(0));
            }

            dJdB.Cache[0] = Tao * dJdB.Cache[0] + ((1 - Tao) * dJdB);
            return (b - Alpha * dJdB.Cache[0]);
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