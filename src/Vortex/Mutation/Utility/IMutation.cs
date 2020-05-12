// Copyright © 2020 Void-Intelligence All Rights Reserved.

namespace Vortex.Mutation.Utility
{
    public interface IMutation
    {
        public double Mutate(double value);

        public EMutationType Type();
    }
}
