// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Optimizer.Utility
{
    public abstract class Optimizer
    {
        public double Alpha { get; set; }
        public Optimizer(OptimizerSettings settings) {}
        public abstract Matrix CalculateDeltaW(Matrix W, Matrix dJdW);
        public abstract Matrix CalculateDeltaB(Matrix b, Matrix dJdb);
        public abstract EOptimizerType Type();
    }

    public abstract class OptimizerSettings
    {
    }
}
