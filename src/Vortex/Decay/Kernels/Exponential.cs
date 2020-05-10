// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Decay.Utility;

namespace Vortex.Decay.Kernels
{
    public class Exponential : BaseDecay
    {
        public Exponential(double decay) : base(decay)
        {
        }

        public override double CalculateAlpha(double alpha)
        {
            return alpha * Math.Exp(-Decay * Epoch);
        }

        public override EDecayType Type()
        {
            return EDecayType.Exponential;
        }
    }
}