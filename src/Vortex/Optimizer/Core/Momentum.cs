// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Momentum : Utility.BaseOptimizer
    {
        private Matrix _vDw;
        private Matrix _vDb;

        /// <summary>
        /// Momentum Parameter
        /// </summary>
        public double Tao { get; set; }

        public Momentum(MomentumSettings settings) : base(settings)
        {
            Tao = settings.Tao;
        }

        public Momentum(double tao = 0.9, double alpha = 0.001) : base(new MomentumSettings(tao, alpha))
        {
        }

        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            if (_vDw == null)
            {
                _vDw = (Alpha * (w.Hadamard(dJdW)));
            }
            else
            {
                _vDw = (Tao * _vDw) + ((1 - Tao) * dJdW);
            }
            return _vDw;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            if (_vDb == null)
            {
                _vDb = (Alpha * (b.Hadamard(dJdB)));
            }
            else
            {
                _vDb = (Tao * _vDb) + ((1 - Tao) * dJdB);
            }
            return _vDb;
        }

        public override EOptimizerType Type() => EOptimizerType.Momentum;
    }

    public sealed class MomentumSettings : OptimizerSettings
    {
        public double Tao { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Momentum;

        public MomentumSettings(double tao = 0.9, double alpha = 0.001) : base(alpha)
        {
            Tao = tao;
        }
    }
}