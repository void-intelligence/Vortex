// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class NadamKernel : BaseOptimizerKernel
    {
        public NadamKernel(Nadam settings) : base(settings)
        {
        }
        public NadamKernel(double alpha = 0.001) : base(new Nadam(alpha))
        {
        }

        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            return null;
        }
        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.Nadam;
    }

    public sealed class Nadam : Utility.Optimizer
    {
        public override EOptimizerType Type() => EOptimizerType.Nadam;

        public Nadam(double alpha = 0.001) : base(alpha)
        {
        }
    }
}
