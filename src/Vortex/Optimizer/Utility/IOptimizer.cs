// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;

namespace Vortex.Optimizer.Utility
{
    public interface IOptimizer
    {
        public Matrix CalculateDelta(Matrix x, Matrix dJdX);
        public EOptimizerType Type();
    }
}