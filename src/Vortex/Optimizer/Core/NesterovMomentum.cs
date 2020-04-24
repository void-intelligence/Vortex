// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class NesterovMomentum : Utility.BaseOptimizer
    {
        public NesterovMomentum(NesterovMomentumSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override string ToString() => Type().ToString();

        public override EOptimizerType Type() => EOptimizerType.NesterovMomentum;

        public override Matrix CalculateDeltaW(Matrix W, Matrix dJdW)
        {
            return null;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdb)
        {
            return null;
        }
    }

    public sealed class NesterovMomentumSettings : OptimizerSettings
    {
        public double Alpha { get; set; }
        public override EOptimizerType Type() => EOptimizerType.NesterovMomentum;
    }
}
