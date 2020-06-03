// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;

namespace Vortex.Metrics.Utility
{
    public interface IMetric
    {
        public double Evaluate(Matrix actual, Matrix expected);
    }
}
