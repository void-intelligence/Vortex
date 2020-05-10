// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Mutation.Utility;

namespace Vortex.Mutation.Kernels
{
    public sealed class NoMutationKernel : BaseMutationKernel
    {
        public override double Mutate(double value)
        {
            return value;
        }

        public override EMutationType Type()
        {
            return EMutationType.NoMutation;
        }
    }

    public sealed class NoMutation : BaseMutation
    {
        public override EMutationType Type()
        {
            return EMutationType.NoMutation;
        }
    }
}
