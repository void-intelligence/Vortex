// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Adam : Utility.Optimizer
    {
        public Adam(AdamSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override string ToString() => Type().ToString();

        public override EOptimizerType Type() => EOptimizerType.Adam;

        public override Matrix CalculateDeltaW(Matrix W, Matrix dJdW)
        {
            return null;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdb)
        {
            return null;
        }
    }

    public sealed class AdamSettings : OptimizerSettings
    {
        public double Alpha { get; set; }
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public double BetaPrimary_T { get; set; }
        public double BetaSecondary_T { get; set; }
        public double Epsilon { get; set; }
    }
}
