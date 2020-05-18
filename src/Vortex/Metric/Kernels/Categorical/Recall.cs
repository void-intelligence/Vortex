// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Metrics.Utility;

namespace Vortex.Metric.Kernels.Categorical
{
    public sealed class Recall : BaseMetric
    {
        public Recall(double threshold = 0.5) : base(threshold)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            var val = 0.0;
            var div = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                if (!(Math.Abs(expected[i, j] - 1.0) < 0.1)) continue;
                div++;
                if (actual[i, j] >= Threshold)
                    val++;
            }

            val /= div;
            if (double.IsNaN(val)) val = 0.0;
            return val;
        }

        public override EMetricType Type()
        {
            return EMetricType.CategoricalRecall;
        }
    }
}