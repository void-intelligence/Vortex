// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Kernels;

namespace Vortex.Optimizer.Utility
{
    public sealed class DefaultOptimizer : GradientDescent
    {
#nullable enable
        public DefaultOptimizer(double alpha = 0.01, IDecay? decay = null) : base(alpha, decay)
        {
        }
#nullable disable

        public override EOptimizerType Type()
        {
            return EOptimizerType.Default;
        }
    }
}