// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Decay.Kernels;
using Vortex.Decay.Utility;

namespace Vortex.Optimizer.Utility
{
    public abstract class BaseOptimizer : IOptimizer
    {
        public IDecay Decay { get; }

        public double Alpha { get; set; }
#nullable enable
        protected BaseOptimizer(double alpha, IDecay? decay = null)
        {
            Alpha = alpha;
            Decay = decay ?? new None();
        }
#nullable disable

        public virtual void ApplyDecay()
        {
            ((BaseDecay)Decay).IncrementEpoch();
            Alpha = Decay.CalculateAlpha(Alpha);
        }

        public abstract Matrix CalculateDelta(Matrix x, Matrix dJdX);
        public abstract EOptimizerType Type();
    }
}
