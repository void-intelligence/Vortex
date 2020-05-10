// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Decay.Utility;

namespace Vortex.Decay.Kernels
{
    public class Multiplication : BaseDecay
    {
        public double Multiplier { get; }
        public double Interval { get; set; }
        public Multiplication(double decay, double interval, double multiplier) : base(decay)
        {
            Multiplier = multiplier;
            Interval = interval;
        }

        public override double CalculateAlpha(double alpha)
        {
            return alpha * Math.Pow(Multiplier, Epoch / Interval);
        }

        public override EDecayType Type()
        {
            return EDecayType.Multiplication;
        }
    }
}