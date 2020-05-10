// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;

namespace Vortex.Mutation.Utility
{
    public abstract class BaseMutation
    {
        public Random Rng { get; }

        protected BaseMutation()
        {
            Rng = new Random();
        }

        public abstract double Mutate(double value);

        public abstract EMutationType Type();
    }
}
