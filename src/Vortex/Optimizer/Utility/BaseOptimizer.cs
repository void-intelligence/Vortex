// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Optimizer.Utility
{
    public abstract class BaseOptimizer
    {
        public double Alpha { get; set; }
        public BaseOptimizer(OptimizerSettings settings) { Alpha = settings.Alpha; }
        public abstract Matrix CalculateDelta(Matrix X, Matrix dJdX);
        public abstract EOptimizerType Type();
    }

    public abstract class OptimizerSettings
    {
        public double Alpha { get; set; }
        public abstract EOptimizerType Type();

        protected OptimizerSettings(double alpha)
        {
            Alpha = alpha;
        }
    }
}
