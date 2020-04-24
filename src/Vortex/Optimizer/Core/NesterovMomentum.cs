// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class NesterovMomentum : Utility.BaseOptimizer
    {
        public NesterovMomentum(NesterovMomentumSettings settings) : base(settings)
        {
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.NesterovMomentum;
    }

    public sealed class NesterovMomentumSettings : OptimizerSettings
    {
        public override EOptimizerType Type() => EOptimizerType.NesterovMomentum;

        public NesterovMomentumSettings(double alpha) : base(alpha)
        {
        }
    }
}
