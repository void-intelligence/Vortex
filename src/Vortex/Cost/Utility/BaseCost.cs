// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Regularization;
using Vortex.Regularization.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Utility
{
    /// <summary>
    /// A cost function is a measure of "how good" a neural network did with respect to it's given training sample and the expected output. 
    /// It also may depend on variables such as weights and biases.
    /// A cost function is a single value, not a vector, because it rates how good the neural network did as a whole.
    /// </summary>
    public abstract class BaseCost
    {
        public BaseCost(CostSettings settings) { }
        
        public abstract double Forward(Matrix Actual, Matrix Expected, int layerCount);

        public abstract Matrix Backward(Matrix Actual, Matrix Expected, int layerCount);

        public abstract ECostType Type();

        public double BatchCost { get; protected set; }

        public virtual void ResetCost() { BatchCost = 0; }
    }

    public abstract class CostSettings { }
}
