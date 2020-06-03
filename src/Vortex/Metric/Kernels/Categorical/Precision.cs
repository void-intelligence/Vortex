// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Core;
using Vortex.Metrics.Utility;

namespace Vortex.Metric.Kernels.Categorical
{
    public sealed class Precision : BaseMetric
    {
        public Precision(double threshold = 0.5) : base(threshold)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            var val = 0.0;
            var div = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                if (!(actual[i, j] >= Threshold)) continue;
                div++;
                if (Math.Abs(expected[i, j] - 1.0) < 0.1)
                    val++;
            }

            val /= div;
            return val;
        }

        public override EMetricType Type()
        {
            return EMetricType.CategoricalPrecision;
        }
    }
}