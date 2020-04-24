// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Momentum : Utility.BaseOptimizer
    {
        public double Tao { get; set; }

        public Momentum(MomentumSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
            Tao = settings.Tao;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.Momentum;
    }

    public sealed class MomentumSettings : OptimizerSettings
    {
        public double Alpha { get; set; }
        public double Tao { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Momentum;
    }
}
