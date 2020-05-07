// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class NesterovMomentumKernel : BaseOptimizerKernel
    {
      

        public NesterovMomentumKernel(NesterovMomentum settings) : base(settings)
        {
        }

        public NesterovMomentumKernel(double alpha = 0.001) : base(new NesterovMomentum(alpha))
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


        public override EOptimizerType Type() => EOptimizerType.NesterovMomentum;
    }

    public sealed class NesterovMomentum : Utility.Optimizer
    {
        public override EOptimizerType Type() => EOptimizerType.NesterovMomentum;

        public NesterovMomentum(double alpha = 0.001) : base(alpha)
        {
        }
    }
}


