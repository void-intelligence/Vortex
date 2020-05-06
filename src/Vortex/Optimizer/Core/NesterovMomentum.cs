// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class NesterovMomentum : Utility.BaseOptimizer
    {
      

        public NesterovMomentum(NesterovMomentumSettings settings) : base(settings)
        {
        }

        public NesterovMomentum(double alpha = 0.001) : base(new NesterovMomentumSettings(alpha))
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

    public sealed class NesterovMomentumSettings : OptimizerSettings
    {
        public override EOptimizerType Type() => EOptimizerType.NesterovMomentum;

        public NesterovMomentumSettings(double alpha = 0.001) : base(alpha)
        {
        }
    }
}


