// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Metrics.Utility;
using Vortex.Cost.Kernels.Regression;

namespace Vortex.Metric.Kernels.Regression
{
    public sealed class RMSLE : BaseMetric
    {
        public RMSLE() : base(0.0)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Math.Sqrt(new MSLE().Evaluate(actual, expected));
        }

        public override EMetricType Type()
        {
            return EMetricType.RegressionRMSE;
        }
    }
}