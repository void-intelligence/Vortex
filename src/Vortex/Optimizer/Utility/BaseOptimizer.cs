// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Optimizer.Utility
{
    public abstract class BaseOptimizer
    {
        public double Alpha { get; set; }
        protected BaseOptimizer(OptimizerSettings settings) { Alpha = settings.Alpha; }
        public abstract Matrix CalculateDeltaW(Matrix w, Matrix dJdW);
        public abstract Matrix CalculateDeltaB(Matrix b, Matrix dJdB);
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
