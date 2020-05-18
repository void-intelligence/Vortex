// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Metrics.Utility;
using Vortex.Cost.Kernels.Regression;

namespace Vortex.Metric.Kernels.Regression
{
    public sealed class RSquared : BaseMetric
    {
        public RSquared() : base(0.0)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            var actualAvg = actual.Average();
            var num = 0.0;
            var denom = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) num += Math.Pow(expected[i, j] - actual[i, j], 2);

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) denom += Math.Pow(expected[i, j] - actualAvg, 2);

            return num / (denom + double.Epsilon);
        }

        public override EMetricType Type()
        {
            return EMetricType.RegressionRSquared;
        }
    }
}