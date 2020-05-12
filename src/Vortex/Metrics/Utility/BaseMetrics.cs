// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Metrics.Utility
{
    public abstract class BaseMetrics : IMetrics
    {
        public double Threshold { get; set; }

        protected BaseMetrics(double threshold)
        {
            Threshold = threshold;
        }

        public abstract double Evaluate(Matrix actual, Matrix expected);
        public abstract EMetricsType Type();
    }
}
