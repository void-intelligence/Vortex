// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Core;
using Vortex.Metrics.Utility;
using Vortex.Cost.Kernels.Regression;

namespace Vortex.Metric.Kernels.Regression
{
    public sealed class CosineSimilarity : BaseMetric
    {
        public CosineSimilarity() : base(0.0)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            var sum = (actual * expected).Sum();
            return sum / Math.Sqrt(expected.Hadamard(expected).Sum()) + Math.Sqrt(actual.Hadamard(actual).Sum());
        }

        public override EMetricType Type()
        {
            return EMetricType.RegressionRMSE;
        }
    }
}