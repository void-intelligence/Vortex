// Copyright © 2020 Void-Intelligence All Rights Reserved.

namespace Vortex.Decay.Utility
{
    public abstract class BaseDecay : IDecay
    {
        public int Epoch { get; private set; }
        public double Decay { get; set; }

        protected BaseDecay(double decay)
        {
            Decay = decay;
            Epoch = 0;
        }

        public void IncrementEpoch()
        {
            Epoch++;
        }

        public abstract double CalculateAlpha(double alpha);

        public abstract EDecayType Type();
    }
}
