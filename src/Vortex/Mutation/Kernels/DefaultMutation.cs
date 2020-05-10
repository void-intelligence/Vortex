// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Mutation.Utility;

namespace Vortex.Mutation.Kernels
{
    public sealed class DefaultMutation : BaseMutation
    {
        public override double Mutate(double value)
        {
            var randomNumber = Rng.Next(0, 100);

            if (randomNumber <= 2f)
            {
                //if 1
                //flip sign of weight
                value *= -1f;
            }
            else if (randomNumber <= 4f)
            {
                //if 2
                // random value between -1 and 1
                value = ((float)Rng.NextDouble() - 0.5) * 2;
            }
            else if (randomNumber <= 6f)
            {
                //if 3
                //randomly increase by 0% to 100%
                var factor = (float)Rng.NextDouble() + 1f;
                value *= factor;
            }
            else if (randomNumber <= 8f)
            {
                //if 4
                //randomly decrease by 0% to 100%
                var factor = (float)Rng.NextDouble();
                value *= factor;
            }
            return value;
        }

        public override EMutationType Type()
        {
            return EMutationType.DefaultMutation;
        }
    }
}
