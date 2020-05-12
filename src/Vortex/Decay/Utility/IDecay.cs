// Copyright © 2020 Void-Intelligence All Rights Reserved.

namespace Vortex.Decay.Utility
{
    public interface IDecay
    {
        public double CalculateAlpha(double alpha);

        public EDecayType Type();
    }
}