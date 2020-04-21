// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Momentum : Utility.Optimizer
    {
        public double Tao { get; set; }

        public Momentum(MomentumSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
            Tao = settings.Tao;
        }

        public override string ToString() => Type().ToString();

        public override EOptimizerType Type() => EOptimizerType.Momentum;

        public override Matrix CalculateDeltaW(Matrix W, Matrix dJdW)
        {
            return null;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdb)
        {
            return null;
        }
    }

    public sealed class MomentumSettings : OptimizerSettings
    {
        public double Alpha { get; set; }
        public double Tao { get; set; }
    }
}
