// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Mutation.Utility;

namespace Vortex.Mutation.Kernels
{
    public sealed class DefaultMutation : BaseMutation
    {
        public override double Mutate(double value)
        {
            var randomNumber = Rng.Next(0, 100);

            if (randomNumber <= 2)
            {
                // Flip sign of weight
                value *= -1;
            }
            else if (randomNumber <= 4f)
            {
                // Between -1 and 1
                value = (Rng.NextDouble() - 0.5) * 2;
            }
            else if (randomNumber <= 6)
            {
                // Increase by 0% to 100%
                value *= (Rng.NextDouble() + 1.0);
            }
            else if (randomNumber <= 8)
            {
                // Decrease by 0% to 100%
                value *= Rng.NextDouble();
            }
            return value;
        }

        public override EMutationType Type()
        {
            return EMutationType.DefaultMutation;
        }
    }
}
