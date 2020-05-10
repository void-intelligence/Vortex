// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Decay.Utility;

namespace Vortex.Decay.Kernels
{
    public class None : BaseDecay
    {
        public None() : base(0)
        {
        }

        public override double CalculateAlpha(double alpha)
        {
            return alpha;
        }

        public override EDecayType Type()
        {
            return EDecayType.None;
        }
    }
}