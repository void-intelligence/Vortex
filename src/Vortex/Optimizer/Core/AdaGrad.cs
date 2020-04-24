// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class AdaGrad : Utility.BaseOptimizer
    {
        public double Epsilon { get; set; }

        public AdaGrad(AdaGradSettings settings): base(settings)
        {
            Epsilon = settings.Epsilon;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }
        
        public override EOptimizerType Type() => EOptimizerType.AdaGrad;
    }

    public sealed class AdaGradSettings : OptimizerSettings
    {
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.AdaGrad;

        public AdaGradSettings(double alpha, double epsilon) : base(alpha)
        {
            Epsilon = epsilon;
        }
    }
}
