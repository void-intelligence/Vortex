// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Decay.Utility;

namespace Vortex.Decay.Kernels
{
    public class Subtraction : BaseDecay
    {
        public double Interval { get; set; }

        public Subtraction(double decay, double interval) : base(decay)
        {
            Interval = Interval;
        }

        public override double CalculateAlpha(double alpha)
        {
            return alpha - Decay * Epoch / Interval;
        }

        public override EDecayType Type()
        {
            return EDecayType.Subtraction;
        }
    }
}