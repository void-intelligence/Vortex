// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Decay.Utility;

namespace Vortex.Decay.Kernels
{
    public class IterationBasedKernel : BaseDecay
    {
        public IterationBasedKernel(double decay) : base(decay)
        {
        }

        public override double CalculateAlpha(double alpha)
        {
            return alpha / (1.0 + Decay * Epoch);
        }

        public override EDecayType Type()
        {
            return EDecayType.IterationBased;
        }
    }
}